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

from utils import parse_experiment_args
from datetime import datetime

import subprocess
import os
import numpy as np

def run_experiment(dataset='gsm8k', prompting_method='dual-prompting', num_samples=200, output_dir='/output/', model="llama3.1:7b", n_shots=8):
    """
    Launches main.py with the given args.

    Args:
        dataset (str, optional): Defaults to 'gsm8k'.
        prompting_method (str, optional): Defaults to 'dual-prompting'.
        num_samples (int, optional): Defaults to 200.
        output_dir (str, optional): Defaults to '/output/'.
        model (str, optional): Defaults to "llama3.1:7b".
    """
    
    # generate a random seed
    seed = np.random.randint(1,9999)
    cmd = f"python main.py --dataset {dataset} --prompting_method {prompting_method} --num_samples {num_samples} --output_dir {output_dir} --model {model} --seed {seed} --n_shots {n_shots}"
    subprocess.run(cmd, shell=True)

def main():
    # unpack args
    args = parse_experiment_args()

    dataset = args.dataset
    model = args.model
    num_trials = args.num_trials
    num_samples = args.num_samples
    n_shots = args.n_shots

    time = datetime.now().strftime(r"%H-%M_%Y-%m-%d")
    folder_name = f"{dataset}_{model}_{time}"
    output_dir = os.path.join("experiments", folder_name)

    for i in range(num_trials):
        for prompting_method in args.prompting_methods:
            print(f"\n\nRunning experiment {i} for {prompting_method} ({n_shots}-shot) on dataset {dataset} and model {model}...")
            run_experiment(dataset, prompting_method, num_samples, output_dir, model, n_shots)
        
    print("All experiments complete.")

    
    print("Generating artifacts...")
    cmd = f"python generate_artifacts.py --exp_dir {output_dir}"
    subprocess.run(cmd, shell=True)
    print("Artifacts generated.")

if __name__ == "__main__":
    main()