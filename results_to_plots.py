"""

Script to take generated experiment json data and return
plots demonstrating a variety of infomration about the results.

"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

EXP_RESULT_SUMMARY_NAME = "exp_results.json"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir",
        default="./experiments/",
        type=str
    )

    return parser.parse_args()


def dir_to_plots(exp_dir: str):
    """
    Takes a experiment directory and generates
    plots for that experiment in the given directory

    Args:
        exp_dir: str = the directory in which we store experiment jsons
    """

    # find experiment files in the given target directory
    all_jsons = [f for f in os.listdir(exp_dir) if f.endswith('.json')]

    # separate the summary from the data
    if EXP_RESULT_SUMMARY_NAME in all_jsons:
        exp_results_path = os.path.join(exp_dir, EXP_RESULT_SUMMARY_NAME)
        all_jsons.remove(EXP_RESULT_SUMMARY_NAME)
    else:
        raise ValueError("Missing experiment results in target directory.")

    all_jsons = [os.path.join(exp_dir, i) for i in all_jsons]

    # print data statistics
    print(f'experiment summary located in {exp_results_path}\n')
    print(f'total experiments parsed: {len(all_jsons)}\n')

    with open(exp_results_path, "r", encoding="utf-8") as f:
        exp_results = json.load(f)
    print(f'Experiment summary: {exp_results}\n')

    with open(all_jsons[0], "r", encoding="utf-8") as f:
        this_exp = json.load(f)
    print(f'Sample experiment data: {this_exp}\n')

    # yank the per-method scores
    exp_results = exp_results['results']
    methods = list(exp_results.keys())
    method_to_accs = {}

    for file in all_jsons:
        with open(file, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
            method = exp_data['prompting_method']
            acc = exp_data['accuracy']

            if method in method_to_accs:
                method_to_accs[method].append(acc)
            else:
                method_to_accs[method] = [acc]

            

    # bar chart w/ mean accs
    means = [exp_results[m][0] for m in methods]
    stdevs = [np.sqrt(exp_results[m][1]) for m in methods]

    plt.figure(figsize=(6,4))
    plt.bar(methods, means, yerr=stdevs, capsize=5, color=['skyblue', 'lightgreen'])
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy by Prompting Method')
    plt.tight_layout()

    out_path = os.path.join(exp_dir, 'mean_accs.png')
    plt.savefig(out_path)
    plt.close()

    # box plot with the data on a per-experiment basis
    plot_data = [method_to_accs[m] for m in methods]
    plt.figure(figsize=(6,4))
    plt.boxplot(plot_data, tick_labels=methods)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution by Prompting Method')
    plt.tight_layout()

    out_path = os.path.join(exp_dir, 'box_plot.png')
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    dir_to_plots(args.exp_dir)
