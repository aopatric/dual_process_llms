"""
This file takes in a set of arguments, runs the experiments
using main.py, and generate experiment artifacts in the target directory:
- a log file that contains the time stamp, arguments, and results
    for the experiment
- a graph comparing the performance of different prompting methods
    on a given dataset
- a table that gives the same information

Args:
    --dataset: the dataset to run the experiments on
    --model: the model to run the experiments on
    --prompting_methods: the prompting methods to run the experiments on
    --num_samples: the number of samples to run the experiments on
    --seed: the seed to run the experiments on
"""

from utils import *

import subprocess

def run_experiment(dataset, model, prompting_method, num_samples, seed, output_dir=None):
    """
    This function essentially calls main.py with the given arguments for
    each prompting method and processes the results
    """
    cmd = f"python main.py --dataset {dataset} --model {model} --prompting_method {prompting_method} --num_samples {num_samples} --seed {seed}"
    if output_dir:
        cmd += f" --output_dir {output_dir}"
    subprocess.run(cmd, shell=True, capture_output=True, text=True)

def main():
    args = parse_experiment_args()

    for prompting_method in args.prompting_methods:
        print(f"Running experiments for {prompting_method}...")
        run_experiment(args.dataset, args.model, prompting_method, args.num_samples, args.seed)

if __name__ == "__main__":
    main()