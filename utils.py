import argparse
import numpy as np
import os

from datasets import load_dataset
from dataclasses import dataclass

"""
Constants

"""

HF_TOKEN = os.getenv("HF_TOKEN")

DATASETS = {
    "gsm8k" : {
        "answer_trigger": "#### ", # gsm8k final answers preceded by this prefix
        "path": "gsm8k",
        "branch": "main",
        "question_field": "question",
        "answer_field": "answer"
    },
    "mathqa": {
        "answer_trigger": "",
        "path": "math_qa",
        "branch": "main",
        "question_field": "Problem",
        "answer_field": "Rationale"
    },
    "math": {
        "answer_trigger": "oxed",  # Verify this trigger
        "path": "competition_math",
        "branch": "main", 
        "question_field": "problem",
        "answer_field": "solution"
    }
}

PROMPTING_METHODS = [
    "chain-of-thought",
    "dual-prompting",
]

"""
Useful functions

"""

# helper for parsing input arguments using argparse library
def parse_input_args():
    parser = argparse.ArgumentParser(description="dual-process reasoning in LLMs")

    # set a random seed here in case we only run from the main script
    seed = np.random.randint(1, 9999)

    parser.add_argument("--dataset", type=str, default="gsm8k", choices=DATASETS.keys())
    parser.add_argument("--prompting_method", type=str, default="dual-prompting", choices=PROMPTING_METHODS)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--model", type=str, default="llama3.1", choices=["llama3.1:70b", "llama3.1", "gpt4-o"])
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--n_shots", type=int, default=8)

    args = parser.parse_args()
    return args

# helper for parsing input arguments for running experiments
def parse_experiment_args():
    exp_parser = argparse.ArgumentParser(description="dual-process reasoning in LLMs -> running experiments")
    exp_parser.add_argument("--dataset", type=str, default="gsm8k", choices=DATASETS.keys())
    exp_parser.add_argument("--prompting_methods", type=str, default=['chain-of-thought', 'dual-prompting'], choices=PROMPTING_METHODS, nargs="+")
    exp_parser.add_argument("--model", type=str, default="llama3.1", choices=["llama3.1:70b", "llama3.1", "gpt4-o"])
    exp_parser.add_argument("--num_samples", type=int, default=200)
    exp_parser.add_argument("--n_shots", type=int, default=8)
    exp_parser.add_argument("--num_trials", type=int, default=10)

    # set a random seed here in case we run from the experiments script
    seed = np.random.randint(1, 9999)
    exp_parser.add_argument("--seed", type=int, default=seed)

    args = exp_parser.parse_args()
    return args

# helper for parsing input arguments for generating plots
def parse_plot_args():
    plot_parser = argparse.ArgumentParser(description="dual-process reasoning in LLMs -> generating plots")
    plot_parser.add_argument("--exp_dir", type=str, default=None)
    args = plot_parser.parse_args()
    return args

# safely load data by the arguments
def safe_load_data(args):
    dset_info = DATASETS[args.dataset]
    n_samples = args.num_samples
    seed = args.seed

    # get a random subset of length n_samples from the requested dataset
    data = load_dataset(
                        dset_info["path"],
                        dset_info["branch"],
                        split="test",
                        trust_remote_code=True
                    ).shuffle(seed=seed)[:n_samples]
    return data

@dataclass
class ExperimentResult:
    prompting_method: str
    accuracy: float
    total_samples: int
    dropped_samples: int
    model_name: str
    dataset_name: str
    seed: int
    n_shots: int