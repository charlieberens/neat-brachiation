import argparse
import os
from parallel_scene_numba import ParallelScene
from neat.population import Population
from neat.config import get_config
from neat.plotting import plot_species

constraints = {
        "velocity": 12,
        "length": (1.25, 2.75),
        "dl": 5,
        "tourque": 12,
        "w": 12,
        "death_distance": -5,
        "hold_time": [.2, 3],
    }

def eval_generation(parallel_scene:ParallelScene, organisms):
    return parallel_scene.evaluate_organisms(organisms)

def run(args):
    model_id = args.model
    config = get_config(args.config)
    max_time = args.t__max_time

    ps = ParallelScene(NUM_THREADS=32, constraints=constraints, dt=.02, max_time=max_time)
    p = Population.from_file(os.path.join(".","{}".format(model_id)), config)

    winner = p.best

    print("Winner Fitness: {}".format(winner.fitness))
    # No clue why I have to do this opposed to making an array of just winner, but very rarely it doesn't work without it
    orgs = [o for o in p.organisms]
    orgs[0] = winner
    ps.evaluate_organisms(orgs, draw=True, filename="winner.gif")

def main():
    argparser = argparse.ArgumentParser(
        description="Evaluate how well a NEAT network gibbons",
    )
    argparser.add_argument(
        "-m",
        "--model",
        help="Model to use (relative to progress directory)",
        required=True,
    )
    argparser.add_argument(
        "-c",
        "--config",
        help="Config file to use",
        default="config.json"
    )
    argparser.add_argument(
        "-t"
        "--max-time",
        type=float,
        default=20,
        help="Max time for a gibbon to gibbon for",
    )
    # argparser.add_argument(
    #     "-p",
    #     "--plot",
    #     action="store_true",
    #     help="Plot the species progress",
    # )

    args = argparser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
