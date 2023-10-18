import argparse
import time
from parallel_scene_numba import ParallelScene
from neat.population import Population
from neat.config import get_config
from neat.reporter import PrintReporter, ProgressReporter, StatReporter, SpeciesStatReporter
import os

# import visualize

"""
Input nodes:
0,1: pos
2,3: v
4: l
5: th
6,7: anchor_dist

Output nodes:
0: dl
1: torque
2: hold
"""


def eval_generation(parallel_scene:ParallelScene, organisms):
    return parallel_scene.evaluate_organisms(organisms)

def run(args):
    model_id = args.model
    initial_number = args.starting_iteration
    iterations = args.iterations
    organism_count = args.organism_count
    batch_size = args.batch_size
    config_file = args.config
    thread_count = args.thread_count
    max_time = args.max_time

    constraints = {
        "velocity": 12,
        "length": (1.25, 2.75),
        "dl": 5,
        "tourque": 12,
        "w": 12,
        "death_distance": -5,
        "hold_time": [.2, 3],
    }
    
    for i in range(iterations):
        print("ROUND", i)
        config = get_config(config_file)

        if initial_number + i * batch_size > 0:
            p = Population.from_file(os.path.join("progress","{}-{}".format(model_id, str(initial_number + i * batch_size))), config)
            if p.id != model_id:
                p.id = model_id
        else:
            model_id = model_id if model_id else str(int(time.time())) 
            p = Population(organism_count, config, id = model_id)

        # Add reporters to show progress in the terminal
        p.add_reporter(PrintReporter(species=True))
        p.add_reporter(ProgressReporter(frequency=batch_size))
        p.add_reporter(StatReporter(stats=["best_fitness", "avg_fitness", "worst_fitness"]))
        p.add_reporter(SpeciesStatReporter())

        ps = ParallelScene(NUM_THREADS=thread_count, constraints=constraints, dt=.02, max_time=max_time)

        p.run(lambda organisms: eval_generation(ps, organisms), organism_count, en_masse=True)


def main():
    argparser = argparse.ArgumentParser(
        description="Train a NEAT network to gibbon",
    )
    argparser.add_argument(
        "-m",
        "--model",
        help="Model to use",
        required=True
    )
    argparser.add_argument(
        "-s",
        "--starting-iteration",
        help="Iteration number to start at",
        type=int,
        default=0
    )
    argparser.add_argument(
        "-i",
        "--iterations",
        help="Number of iterations to run",
        type=int,
        default=1024
    )
    argparser.add_argument(
        "-o",
        "--organism-count",
        help="Number of organisms to train at a time",
        type=int,
        default=2048
    )
    argparser.add_argument(
        "--batch-size",
        help="Number of epochs between saves",
        type=int,
        default=64
    )
    argparser.add_argument(
        "-c",
        "--config",
        help="Config file to use",
        default="config.json"
    )
    argparser.add_argument(
        "--thread-count",
        help="Number of threads to use",
        type=int,
        default=32
    )
    argparser.add_argument(
        "-t",
        "--max-time",
        help="Max time to let a gibbon gibbon",
        type=int,
        default=20
    )
    args = argparser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
