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
import uuid

def run_experiment(dataset, prompting_method, num_samples, seed, output_dir='/default/', model="llama3.1:7b"):
    """
    This function essentially calls main.py with the given arguments for
    each prompting method and processes the results
    """
    cmd = f"python main.py --dataset {dataset} --prompting_method {prompting_method} --num_samples {num_samples} --seed {seed} --output_dir {output_dir} --model {model}"
    subprocess.run(cmd, shell=True)

def main():
    args = parse_experiment_args()
    output_dir = os.path.join("experiments", str(uuid.uuid4()))

    for prompting_method in args.prompting_methods:
        print(f"Running experiment for {prompting_method} on dataset {args.dataset} and model {args.model}...")
        run_experiment(args.dataset, prompting_method, args.num_samples, args.seed, output_dir, args.model)
    
    print("All experiments complete.")

    print("Generating plots...")
    cmd = f"python generate_plots.py --exp_dir {output_dir}"
    subprocess.run(cmd, shell=True)
    print("Plots generated.")

if __name__ == "__main__":
    main()