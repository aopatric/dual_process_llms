"""
This script generates plots from a set of given .json
experiment results.
"""

import json
import matplotlib.pyplot as plt

from utils import *

def main():
    args = parse_plot_args()
    print(args)
    plot_results(args.exp_dir)

def plot_results(exp_dir):
    # load all the json files in exp_dir
    json_files = [f for f in os.listdir(exp_dir) if f.endswith('.json')]
    
    # list the json files in the target directory for the user 
    print(f"Found {len(json_files)} json files in '{exp_dir}'")

    exp_results = []
    for json_file in json_files:
        with open(os.path.join(exp_dir, json_file), 'r') as f:
            exp_results.append(json.load(f))
    
    # verify the folder represents a valid experiment, i.e. just the same dataset for now
    dataset = exp_results[0]["dataset_name"]
    model = exp_results[0]["model_name"]

    accs = []
    prompting_methods = []
    for exp_result in exp_results:
        if exp_result["dataset_name"] != dataset or exp_result["model_name"] != model:
            raise ValueError(f"Experiment directory is not a valid experiment...")

        accs.append(exp_result["accuracy"])
        prompting_methods.append(exp_result["prompting_method"])

    # plot the results
    bar_plot(accs, prompting_methods, f"Performance of prompting methods on {dataset} using {model}", "Accuracy", exp_dir, y_range=(0, 1))

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
    plt.savefig(os.path.join(exp_dir, f"exp_plot.png"))

if __name__ == "__main__":
    main()
