"""
This script generates plots from a set of given .json
experiment results.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import parse_plot_args

def main():
    args = parse_plot_args()
    print(args)
    process_exp_dir(args.exp_dir)

def process_exp_dir(exp_dir):
    """
    Process the experiment directory to generate the raw graphs for each experiment

    Args:
        exp_dir (str): path to the experiment directory
    """
    # load all available experiment json files the given directory
    all_available_jsons = [f for f in os.listdir(exp_dir) if f.endswith('.json')]
    print(f"Found {len(all_available_jsons)} experiment jsons in the given directory...")

    exp_results = []
    for json_file in all_available_jsons:
        with open(os.path.join(exp_dir, json_file), 'r') as f:
            exp_results.append(json.load(f))

    first_res = exp_results[0]
    dataset, model, num_samples = first_res["dataset_name"], first_res["model_name"], first_res["total_samples"]

    # first, need to calculate the average scores for each prompting method
    results = []
    for result in exp_results:
        if result["dataset_name"] != dataset or result["model_name"] != model or result["total_samples"] != num_samples:
            raise ValueError("Provided directory is not a valid experiment.")
        
        method = result["prompting_method"]
        acc = result["accuracy"]

        results.append((method, acc))
    
    method_to_accs = {}

    for result in results:
        method, acc = result
        if method in method_to_accs:
            method_to_accs[method].append(acc)
        else:
            method_to_accs[method] = [acc]
    
    method_to_stats = {}
    for method in method_to_accs:
        avg_acc = np.mean(method_to_accs[method])
        var = np.var(method_to_accs[method])
        method_to_stats[method] = (avg_acc, var)
        print(f"Mean accuracy for {method} on {dataset} w/ {model}: {avg_acc} (variance {var})")

    output = {
        "dataset": dataset,
        "model": model,
        "num_trials": len(exp_results) // 2, # NOTE: hard coded for now since only have two types of trials
        "num_samples": num_samples,
        "results": method_to_stats,
    }

    out_dir = os.path.join(exp_dir, "exp_results.json")
    with open(out_dir, 'w') as f:
        json.dump(output, f)
    

def bar_plot(values, labels, title, ylabel, exp_dir, y_range=None):
    """
    make a bar chart of the values based on a common dataset
    and label with both model names and prompting methods
    """
    plt.figure(figsize=(10, 5))
    if y_range:
        plt.ylim(y_range)
    plt.bar(labels, values)
    plt.xlabel("Prompting Method")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    # save the plot to the experiment directory
    plt.savefig(os.path.join(exp_dir, "exp_plot.png"))

if __name__ == "__main__":
    main()
