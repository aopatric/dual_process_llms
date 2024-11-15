"""
    Useful functions that make life easier for the rest of the code.
"""

"""
Imports

"""
import argparse

from datasets import load_dataset

"""
Constants

"""

SUPPORTED_DATASETS = [
    "gsm8k",
]

SUPPORTED_MODELS = [
    "api",
]

PROMPTING_METHODS = [
    "default",
]

# unfortunately have to map which one we want to use for ones with multiple splits
CONFIG_MAP = {
    "gsm8k": "main"
}

"""
Useful functions

"""

# helper for parsing input arguments using argparse library
def parse_input_args():
    parser = argparse.ArgumentParser(description="dual-process reasoning in LLMs")

    # adding arguments
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model", type=str, default="api", choices=SUPPORTED_MODELS)
    parser.add_argument("--prompting_method", type=str, default="default", choices=PROMPTING_METHODS)
    parser.add_argument("--num_samples", type=int, default=10)

    args = parser.parse_args()

    return args

# safely load data by only the name, returns all splits of main branch
def safe_load_data(dataset):
    if dataset in CONFIG_MAP:
        train = load_dataset(dataset, CONFIG_MAP[dataset], split="train")
        test = load_dataset(dataset, CONFIG_MAP[dataset], split="test")
    else:
        train = load_dataset(dataset, split="train")
        test = load_dataset(dataset, split="test")

    return (train, test)
