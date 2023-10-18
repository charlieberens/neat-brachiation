import io
import math
import sys
import time
from tkinter import Canvas, Tk
import matplotlib
import numpy as np
import os
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
# os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"
from numba import jit, cuda, float32
from neat.organism import Organism

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image, ImageDraw, ImageTk

WINDOW_SIZE = [500, 500]
RENDER_SCALE = 15

"""
Buffers:
    input:
        arrays:
            nodes_1d
            in_connections_1d
            out_connections_1d
            weights_1d
            node_start_indicies
            connection_start_indicies                    
    
    output:
        fitnesses


CUDA evaluation layout
----------------------
    # Pre-evaluation
        # Set each array to the correct start point;
        nodes_start_index = node_start_indicies[<GID>]
        nodes_1d += nodes_start_index;

        connections_start_index = connection_start_indicies[<GID>]
        in_connections_1d += connections_start_index;
        out_connections_1d += connections_start_index;
        weights_1d += connections_start_index;

        # Get the number of nodes
        int i = 0;
        node_count = 0;
        while(nodes_1d[i]!=-2){
            if(nodes_1d[i]!=-1){
                node_count++;
            }
            i++;                
        }

        # Create node value array
        float node_values[node_count];

        # Create an output array
        float output[output_size];

        evaluate(output)



    # Evaluation
        # Zero all node_values
        for(int i=0; i<node_count; i++){
            node_values[i] = 0;
        }

        # Set input_node_values
        node_values[0] == a_0;
        node_values[1] == a_1;
        ...
        node_values[n] == a_n;

        # Do the rest
        int node_index = 0;
        int connection_index = 0;

        while(nodes_1d[node_index]!=-2){
            # Apply activation function to each node in the layer
            while(nodes_1d[node_index]!=-1){
                int node_number = nodes_1d[node_index]; 
                node_values[node_number] = activation_function(node_values[node_number]);
                node_index++;
            }
            node_index++;

            # Apply the layer's connections - This should reach -2 before nodes_id[node_index] does so.
            while(in_connections_1d[connection_index]!=-1 && out_connections_1d[connection_index]!=-2){
                int in_node_number = in_connections_1d[connection_index];
                int out_node_number = out_connections_1d[connection_index];
                int weight = weights_1d[connection_index];

                node_values[out_node_number] += node_values[in_node_number] * weight;
            }
        }

        # Set output
        node_values_output_point = node_count - output_size;
        for(int i = 0; i < output_size; i++){
            output[i] = node_values[node_values_output_point + i];
        }
    
    # Output buffer
    fitnesses[<GID>] = fitness;
"""

@cuda.jit()
def do_the_cudaing_gibbon(
                        nodes_1d, 
                        node_start_indicies,
                        node_values_arrays,
                        node_counts,
                        in_connections_1d,
                        out_connections_1d,
                        weights_1d,
                        connection_start_indicies,
                        fitnesses,
                        c_velocity,
                        c_length,
                        c_dl,
                        c_tourque,
                        c_w,
                        max_iterations,
                        death_distance,
                        c_hold_time,
                        dt,
                        xs,
                        ys,
                        hxs,
                        hys,
):
    """
        Prepare the data, and send it to the GPU

        input:
            model:
                nodes_1d: int64[]
                node_start_indicies: uint64[]
                node_values_arrays: float64[][]
                node_counts: int64[]
                in_connections_1d: int64[] 
                out_connections_1d: int64[]
                weights_1d: float64[]
                connection_start_indicies: uint64[]
            output:
                fitnesses: float64[]
            constraints:
                input:
                    c_velocity: float64
                    c_length: [float64, float64]
                    c_hold_time: [float64, float64]
                output:
                    c_dl: float64
                    c_tourque: float64
                other:
                    c_w: float64
            meta:
                max_iterations: int64
                death_distance: float64
                dt: float64

        intermediate:
            gibbon input:
                0,1 v: (float64, float64)
                2 length: float64 - preferred length of spring
                3, 4 hand_pos: (float64, float64)
                5, 6 anchor_delta_1: (float64, float64)
                7, 8 anchor_delta_2: (float64, float64)
                9 hold_time: float64
            gibbon output:
                dl: float64
                tourque: float64
                grasp: bool - whether to attempt to hold on to something
            """
    # Compute flattened index inside the array, equivalent to
    # pos = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    thread_block_id = cuda.grid(1)

    if thread_block_id < node_start_indicies.size:
        # Initialize the model
        # Set each array to the correct start point;
        nodes_start_index = node_start_indicies[thread_block_id]
        connections_start_index = connection_start_indicies[thread_block_id]

        node_count = node_counts[thread_block_id]
        node_values = node_values_arrays[thread_block_id]

        output = cuda.local.array(3, dtype=np.float64)
        # Don't forget the bias node
        input = cuda.local.array(11, dtype=np.float64)

        # Constants
        SEED = 1
        SPACING = 8
        HOLD_RADIUS = 1
        GIBBON_RADIUS = .1
        # HOLD_RADIUS = 2
        # GIBBON_RADIUS = .1

        K = 250
        DAMPENING = 10

        # Initialize the gibbon
        v = cuda.local.array(2, dtype=np.float64)
        v[0] = 0
        v[1] = 0
        pos = cuda.local.array(2, dtype=np.float64)
        pos[0] = -1
        pos[1] = 0
        anchor = cuda.local.array(2, dtype=np.float64)
        anchor[0] = 0
        anchor[1] = 0
        hand_pos = cuda.local.array(2, dtype=np.float64)
        hand_pos[0] = 0
        hand_pos[1] = 0
        length = 1
        real_length = 1
        holding = True
        hold_time = 0
        w = 0
        theta = math.atan2(hand_pos[1] - pos[1], hand_pos[0] - pos[0])

        fitness = 0
        glunch = 0

        max_x = 0
        max_y = 0
        max_anchor_index = 0
        
        for i in range(max_iterations):
            # Update the fitness
            ai = get_anchor_index(anchor, SPACING)

            if pos[0] > max_x:
                max_x = pos[0]
            if pos[1] > max_y:
                max_y = pos[1]
            if ai > max_anchor_index:
                max_anchor_index = ai

            if max_x + 4 * max_anchor_index > fitness:
                fitness = max_x + 4 * max_anchor_index

            t = i * dt
            # Velocity
            input[0] = (v[0] + c_velocity) / (2 * c_velocity)
            input[1] = (v[1] + c_velocity) / ( 2 * c_velocity)
            
            # Length
            input[2] = (length - c_length[0]) / (c_length[1] - c_length[0])
            
            # Hand Position
            input[3] = (hand_pos[0] - pos[0] + c_length[1]) / (2 * c_length[1])
            input[4] = (hand_pos[1] - pos[1] + c_length[1]) / (2 * c_length[1])

            # Anchor Positions
            anchor_1, anchor_2 = anchor_oracle(pos, holding, anchor, 0, SPACING)
            input[5] = (anchor_1[0] - pos[0]) / (SPACING)
            input[6] = (anchor_1[1] - pos[1]) / (SPACING)
            input[7] = (anchor_2[0] - pos[0]) / (SPACING)
            input[8] = (anchor_2[1] - pos[1]) / (SPACING)

            input[9] = hold_time / (c_hold_time[1])
            input[10] = 1

            evaluate(input, 11, nodes_1d, nodes_start_index, in_connections_1d, out_connections_1d, connections_start_index, weights_1d, node_values, node_count, output, 3)

            dl = (output[0] + c_dl) / (2 * c_dl)
            tourque = (output[1] + c_tourque) / (2 * c_tourque)
            grasp = output[2] > .5

            # fitness = get_anchor_index(anchor, SPACING)

            length += dl * dt
            length = clamp(length, c_length[0], c_length[1])

            if holding:
                hold_time += dt
            else:
                hold_time = 0


            # Kill the gibbon if it tries to hold on to something for too long or too short
            if (holding and hold_time < c_hold_time[0] and not grasp) or (holding and hold_time > c_hold_time[1] and grasp):
                # glunch = 200
                break

            # Have the gibbon hold on if it is close enough
            if not holding and grasp:
                anchor_1, anchor_2 = anchor_oracle(pos, holding, anchor, 0, SPACING)
                if anchor_1[0] == 2:
                    if pos[0] > 1:
                        if pos[1] > -1:
                            glunch = pos[1]
                if in_hold_box(hand_pos, anchor_1, HOLD_RADIUS):
                    holding = True
                    anchor[0] = anchor_1[0]
                    anchor[1] = anchor_1[1]
                    glunch = get_anchor_index(anchor, SPACING)
                elif in_hold_box(hand_pos, anchor_2, HOLD_RADIUS):
                    holding = True
                    anchor[0] = anchor_2[0]
                    anchor[1] = anchor_2[1]
                    glunch = get_anchor_index(anchor, SPACING)
                else:
                    pass
                    # OPTIONAL - DIE
                
                # Kill the gibbon if it is inside the anchor
                if distance(pos, anchor_1) < (HOLD_RADIUS + GIBBON_RADIUS):
                    break

            # Have the gibbon let go
            if holding and not grasp:
                holding = False
                real_length = distance(pos, hand_pos)
            
            # Update velocities
            f = cuda.local.array(2, dtype=np.float64)
            if holding:
                # DO PENDULUM PHYSICS
                f[0] = 0
                f[1] = 0
                update_with_holding_forces(f, anchor, pos, v, length, dt, K, DAMPENING)
                v[0] += f[0] * dt
                v[1] += f[1] * dt

                # Calculate hand position theta
                old_th = theta
                theta = math.atan2(hand_pos[1] - pos[1], hand_pos[0] - pos[0])
                w = (theta - old_th) / dt
            else:
                f[0] = 0
                f[1] = -9.8
                v[1] += f[1] * dt

                # hand_pos[0] += v[0] * dt
                # hand_pos[1] += v[1] * dt


            # Update the position
            pos[0] += v[0] * dt
            pos[1] += v[1] * dt

            if not holding:
                # Calculate hand position theta
                theta = math.atan2(hand_pos[1] - pos[1], hand_pos[0] - pos[0])
                w += tourque * dt
                w = clamp(w, -c_w, c_w)
                theta += w * dt

                real_length += .25 * (length - real_length)

                hand_pos[0] = pos[0] + math.cos(theta) * real_length
                hand_pos[1] = pos[1] + math.sin(theta) * real_length


            # Clamp the velocity
            v[0] = clamp(v[0], -c_velocity, c_velocity)
            v[1] = clamp(v[1], -c_velocity, c_velocity)

            # Kill the gibbon if it goes too far down
            if pos[1] < death_distance:
                break

            xs[max_iterations * thread_block_id + i] = pos[0]
            ys[max_iterations * thread_block_id + i] = pos[1]
            hxs[max_iterations * thread_block_id + i] = hand_pos[0]
            hys[max_iterations * thread_block_id + i] = hand_pos[1]

        fitnesses[thread_block_id] = fitness
        node_counts[thread_block_id] = glunch
        

@cuda.jit(device=True)
def update_with_spring_force(f, anchor, pos, v, length, K, DAMPENING):
    # TODO - Handle 0 case
    dist = distance(anchor, pos)
    distance_norm = cuda.local.array(2, dtype=np.float64)
    if dist != 0:
        distance_norm[0] = (pos[0] - anchor[0]) / dist
        distance_norm[1] = (pos[1] - anchor[1]) / dist
    else:
        speed = distance(v, (0, 0))
        distance_norm[0] = -v[0] / speed
        distance_norm[1] = -v[1] / speed

    f_mag = 0
    # The spring only applies force if it is stretched
    if dist > length:
        f_mag = -K * (dist - length) - DAMPENING * (v[0] * distance_norm[0] + v[1] * distance_norm[1])
    
    f[0] += f_mag * distance_norm[0]
    f[1] += f_mag * distance_norm[1]


@cuda.jit(device=True)
def update_with_pendulum_force(f, anchor, g_pos, v, length, dt, K, DAMPENING):
    f[0] = 0
    f[1] = -9.8

    pos = cuda.local.array(2, dtype=np.float64)
    pos[0] = g_pos[0] + dt * v[0]
    pos[1] = g_pos[1] + dt * v[1]

    update_with_spring_force(f, anchor, pos, v, length, K, DAMPENING)

@cuda.jit(device=True)
def update_with_holding_forces(f, anchor, pos, v, length, dt, K, DAMPENING):
    temp_v = cuda.local.array(2, dtype=np.float64)
    temp_v[0] = v[0]
    temp_v[1] = v[1]

    # Do the runge kutta integration
    update_with_pendulum_force( f, anchor, pos, temp_v, length, 0, K, DAMPENING)
    k1_0 = f[0]
    k1_1 = f[1]

    temp_v[0] = v[0] + dt * k1_0 / 2
    temp_v[1] = v[1] + dt * k1_1 / 2
    update_with_pendulum_force( f, anchor, pos, temp_v, length, dt / 2, K, DAMPENING)
    k2_0 = f[0]
    k2_1 = f[1]

    temp_v[0] = v[0] + dt * k2_0 / 2
    temp_v[1] = v[1] + dt * k2_1 / 2
    update_with_pendulum_force( f, anchor, pos, temp_v, length, dt / 2, K, DAMPENING)
    k3_0 = f[0]
    k3_1 = f[1]

    temp_v[0] = v[0] + dt * k3_0
    temp_v[1] = v[1] + dt * k3_1
    update_with_pendulum_force( f, anchor, pos, temp_v, length, dt, K, DAMPENING)
    k4_0 = f[0]
    k4_1 = f[1]

    f[0] = (k1_0 + 2 * k2_0 + 2 * k3_0 + k4_0) / 6
    f[1] = (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6

@cuda.jit(device=True)
def in_hold_box(hand_pos, anchor_1, HOLD_RADIUS):
    return abs(hand_pos[0] - anchor_1[0]) < HOLD_RADIUS and abs(hand_pos[1] - anchor_1[1]) < HOLD_RADIUS

@cuda.jit(device=True)
def clamp(x, min, max):
    if x < min:
        return min
    if x > max:
        return max
    return x

@cuda.jit(device=True)
def anchor_oracle(pos, holding, current_anchor, seed, SPACING):
    candidate_1 = cuda.local.array(2, dtype=np.float64)
    candidate_1[0] = pos[0] // SPACING * SPACING + SPACING
    candidate_1[1] = 0

    candidate_2 = cuda.local.array(2, dtype=np.float64)
    candidate_2[0] = pos[0] // SPACING * SPACING + 2 * SPACING
    candidate_2[1] = 0

    candidate_3 = cuda.local.array(2, dtype=np.float64)
    candidate_3[0] = pos[0] // SPACING * SPACING + 3 * SPACING
    candidate_3[1] = 0

    if holding:
        if candidate_1[0] == current_anchor[0]:
            return candidate_2, candidate_3
    return candidate_1, candidate_2

@cuda.jit(device=True)
def get_anchor_index(anchor, SPACING):
    return anchor[0] // SPACING

@cuda.jit(device=True)
def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


@cuda.jit(device=True)
def evaluate(input, input_size, nodes_1d,nodes_start_index, in_connections_1d, out_connections_1d, connections_start_index, weights_1d, node_values, node_count, output, output_size):
    i = 0
    for i in range(node_count):
        node_values[i] = 0

    # Set input_node_values
    for i in range(input_size):
        node_values[i] = input[i]

    # Do the rest
    node_index = 0
    connection_index = 0

    while nodes_1d[node_index+nodes_start_index]!=-2:
        # Apply activation function to each node in the layer
        while nodes_1d[node_index+nodes_start_index]!=-1:
            node_number = nodes_1d[node_index+nodes_start_index] 
            node_values[node_number] = modified_sigmoid(node_values[node_number])
            node_index += 1
        node_index +=1

        # Apply the layer's connections - This should reach -2 before nodes_id[node_index] does so.
        while in_connections_1d[connection_index+connections_start_index]!=-1 and in_connections_1d[connection_index+connections_start_index]!=-2:
            in_node_number = in_connections_1d[connection_index+connections_start_index]
            out_node_number = out_connections_1d[connection_index+connections_start_index]
            weight = weights_1d[connection_index+connections_start_index]

            node_values[out_node_number] += node_values[in_node_number] * weight
            connection_index += 1
        connection_index += 1

    for i in range(output_size):
        node_number = nodes_1d[node_index-1-output_size+i+nodes_start_index] 
        output[i] = node_values[node_number]


@cuda.jit(device=True)
def modified_sigmoid(x):
    return 1 / (1 + math.exp(-4.9 * x))


class ParallelScene:
    def __init__(self, NUM_THREADS, constraints, dt, max_time):
        """
        self.constraints: {
        }
        """
        self.constraints = constraints
        self.dt = dt
        self.max_iterations = int(max_time / dt)
        self.frames = []

        self.NUM_THREADS = NUM_THREADS

    def evaluate_organisms(self, organisms, draw=False, filename=None):
        t = time.time()
        structures = [o.get_network_structure() for o in organisms]

        connection_layerses, node_layerses = ([i for i, j in structures], [j for i, j in structures])

        # Convert lists of layers of nodes to lists of nodes (with -1 as a delimiter)
        node_list_list = []
        node_counts = np.zeros(len(organisms), dtype=np.int32)
        for i, node_layers in enumerate(node_layerses):
            node_list = []
            for node_layer in node_layers:
                node_list.extend(node_layer)
                node_counts[i] += len(node_layer)
                node_list.append(-1)
            node_list_list.append(node_list)
        
        # Convert lists of layers of connections to lists of connections (with -1 as a delimiter)
        connection_input_list_list = []
        connection_output_list_list = []
        connection_weight_list_list = []

        for connection_layers in connection_layerses:
            connection_input_list = []
            connection_output_list = []
            connection_weight_list = []
            for connection_layer in connection_layers:
                for connection in connection_layer:
                    connection_input_list.append(connection[0])
                    connection_output_list.append(connection[1])
                    connection_weight_list.append(connection[2])
                connection_input_list.append(-1)
                connection_output_list.append(-1)
                connection_weight_list.append(-1)
            connection_input_list_list.append(connection_input_list)
            connection_output_list_list.append(connection_output_list)
            connection_weight_list_list.append(connection_weight_list)

        # We need this these to be 1d arrays to do so, we need to know the length of each list. We will use -2 as a delimiter
        nodes_1d = []
        node_start_indicies = []

        i = 0
        for node_list in node_list_list:
            node_start_indicies.append(i)
            nodes_1d.extend(node_list)
            nodes_1d.append(-2)   
            i = len(nodes_1d)
        
        in_connections_1d = []
        out_connections_1d = []
        weights_1d = []
        connection_start_indicies = []

        n = 0
        for i in range(len(connection_input_list_list)):
            connection_start_indicies.append(n)
            in_connections_1d.extend(connection_input_list_list[i])
            out_connections_1d.extend(connection_output_list_list[i])
            weights_1d.extend(connection_weight_list_list[i])
            in_connections_1d.append(-2)
            out_connections_1d.append(-2)
            weights_1d.append(-2)
            n = len(in_connections_1d)
        
        nodes_1d = np.array(nodes_1d, dtype=np.int32)

        in_connections_1d = np.array(in_connections_1d, dtype=np.int32)
        out_connections_1d = np.array(out_connections_1d, dtype=np.int32)
        weights_1d = np.array(weights_1d, dtype=np.float32)
        

        n = len(organisms)
        node_start_indicies = np.array(node_start_indicies, dtype=np.uint32)
        connection_start_indicies = np.array(connection_start_indicies, dtype=np.uint32)
        node_values_arrays = np.zeros((n, max(node_counts)), dtype=np.float64)
        fitnesses = np.zeros(n, dtype=np.float64)

        c_velocity = self.constraints["velocity"]
        c_length = np.array(self.constraints["length"], dtype=np.float64)
        c_dl = self.constraints["dl"]
        c_tourque = self.constraints["tourque"]
        c_w = self.constraints["w"]
        max_iterations = self.max_iterations
        c_death_distance = self.constraints["death_distance"]
        c_hold_time = np.array(self.constraints["hold_time"], dtype=np.float64)
        dt = self.dt

        x_s = np.zeros(n * max_iterations, dtype=np.float64)
        y_s = np.zeros(n * max_iterations, dtype=np.float64)
        hx_s = np.zeros(n * max_iterations, dtype=np.float64)
        hy_s = np.zeros(n * max_iterations, dtype=np.float64)

        data_time = time.time() - t
        t = time.time()

        do_the_cudaing_gibbon[n // self.NUM_THREADS + 1, self.NUM_THREADS](
            nodes_1d,
            node_start_indicies,
            node_values_arrays,
            node_counts,
            in_connections_1d,
            out_connections_1d,
            weights_1d,
            connection_start_indicies,
            fitnesses,
            c_velocity,
            c_length,
            c_dl,
            c_tourque,
            c_w,
            int(max_iterations),
            c_death_distance,
            c_hold_time,
            dt,
            x_s,
            y_s,
            hx_s,
            hy_s)

        cuda_time = time.time() - t
        
        if draw:
            # print(max(fitnesses))
            x_subset = x_s.reshape((n, max_iterations))[0]
            y_subset = y_s.reshape((n, max_iterations))[0]

            hx_subset = hx_s.reshape((n, max_iterations))[0]
            hy_subset = hy_s.reshape((n, max_iterations))[0]

            self.render_trial(x_subset, y_subset, hx_subset, hy_subset)

        if os.environ.get("DEBUG"):
            print("Data time: {:2.4f}s - CUDA time : {:2.4f}s".format(data_time, cuda_time))

        return fitnesses

    def offset_y(self, a):
        return WINDOW_SIZE[1] / 2 - a * RENDER_SCALE

    def offset_x(self, a, x_pos):
        return WINDOW_SIZE[0] / 2 + (a - x_pos) * RENDER_SCALE

    def anchor_oracle(self,x, seed, spacing):
        candidate_1 = [(x // spacing) * spacing + spacing, 0]
        candidate_2 = [(x // spacing) * spacing + 2 * spacing, 0]
        candidate_3 = [(x // spacing) * spacing + 3 * spacing,0]

        return candidate_1, candidate_2, candidate_3


    def render_trial(self, x_pos_arr, y_pos_arr, hx_pos_arr, hy_pos_arr):
        gibbon_pos = (x_pos_arr[0], y_pos_arr[0])
        hand_pos = (hx_pos_arr[0], hy_pos_arr[0])

        MASS_R = .25  * RENDER_SCALE
        SPACING = 8
        HOLD_RADIUS = 1 * RENDER_SCALE

        # After it is called once, the update method will be automatically called every delay milliseconds
        def update(in_gibbon_pos, in_hand_pos):
            # Create a pillow image
            img = Image.new('RGB', (WINDOW_SIZE[0], WINDOW_SIZE[1]), color = (255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Draw a circle on the pillow image

            gibbon_pos = (self.offset_x(in_gibbon_pos[0], in_gibbon_pos[0]), self.offset_y(in_gibbon_pos[1]))
            draw.ellipse([gibbon_pos[0] - MASS_R, gibbon_pos[1] - MASS_R, gibbon_pos[0] + MASS_R, gibbon_pos[1] + MASS_R], fill = (0, 0, 0), outline = (0, 0, 0))

            hand_pos = (self.offset_x(in_hand_pos[0], in_gibbon_pos[0]), self.offset_y(in_hand_pos[1]))
            draw.line([gibbon_pos[0], gibbon_pos[1], hand_pos[0], hand_pos[1]], fill = (0, 0, 0), width = 1)

            anchor_1, anchor_2, anchor_3 = self.anchor_oracle(in_gibbon_pos[0], 0, SPACING)
            anchor_1 = (self.offset_x(anchor_1[0], in_gibbon_pos[0]), self.offset_y(anchor_1[1]))
            anchor_2 = (self.offset_x(anchor_2[0], in_gibbon_pos[0]), self.offset_y(anchor_2[1]))
            anchor_3 = (self.offset_x(anchor_3[0], in_gibbon_pos[0]), self.offset_y(anchor_3[1]))

            draw.ellipse([anchor_1[0] - HOLD_RADIUS, anchor_1[1] - HOLD_RADIUS, anchor_1[0] + HOLD_RADIUS, anchor_1[1] + HOLD_RADIUS], fill = (54, 50, 150), outline = (50, 50, 150))
            draw.ellipse([anchor_2[0] - HOLD_RADIUS, anchor_2[1] - HOLD_RADIUS, anchor_2[0] + HOLD_RADIUS, anchor_2[1] + HOLD_RADIUS], fill = (50, 50, 150), outline = (50, 50, 150))
            draw.ellipse([anchor_3[0] - HOLD_RADIUS, anchor_3[1] - HOLD_RADIUS, anchor_3[0] + HOLD_RADIUS, anchor_3[1] + HOLD_RADIUS], fill = (50, 50, 150), outline = (50, 50, 150))

            self.frames.append(img)

        for i in range(len(x_pos_arr)-1, 0, -1):
            if x_pos_arr[i] == 0:
                x_pos_arr = x_pos_arr[:i]
                y_pos_arr = y_pos_arr[:i]
                hx_pos_arr = hx_pos_arr[:i]
                hy_pos_arr = hy_pos_arr[:i]
            else:
                break


        for x_pos, y_pos, hx_pos, hy_pos in zip(x_pos_arr, y_pos_arr, hx_pos_arr, hy_pos_arr):
            gibbon_pos = (x_pos, y_pos)
            hand_pos = (hx_pos, hy_pos)
            update(gibbon_pos,hand_pos)
            # time.sleep(self.dt)
        

        self.frames[0].save('winner.gif', format='GIF', append_images=self.frames[1:], save_all=True, duration=len(x_pos_arr) * self.dt *.9, loop=0)

        
