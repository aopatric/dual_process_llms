"""
    Useful functions that make life easier for the rest of the code.
"""

"""
Imports

"""
import argparse
import os

from datasets import load_dataset
from dataclasses import dataclass

"""
Constants

"""

# make sure this is set
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

MODELS = {
    "gpt3": {
        "type": "api",
        "name": "babbage-002",
        "api_type": "completion"
    },
    "davinci": {
        "type": "api",
        "name": "davinci-002",
        "api_type": "completion"
    },
    "gpt4": {
        "type": "api",
        "name": "gpt-4",
        "api_type": "chat"
    },
    # disabled b/c the endpoint is different
    "o1mini": {
        "type": "api",
        "name": "o1-mini-2024-09-12",
        "api_type": "chat-o1"
    },
    "tinyllama": {
        "type": "local",
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    # mistral is broken atm, not sure what's wrong, the config file does not get recognized
    "mistral7b": {
        "type": "local",
        "name": "mistralai/Mistral-7B-v0.1",
        "tokenizer": "mistralai/Mistral-7B-v0.1"
    },
    "falcon7b": {
        "type": "local",
        "name": "tiiuae/falcon-7b",
        "tokenizer": "tiiuae/falcon-7b"
    },
    # says we can't connect to huggingface? but other models work
    "llama2_7b": {
        "type": "local",
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "tokenizer": "meta-llama/Llama-2-7b-chat-hf"
    }
}

# exclude broken models from support
DISABLED_MODELS = {"o1mini", "mistral7b", "llama2_7b"}
SUPPORTED_MODELS = [key for key in MODELS.keys() if key not in DISABLED_MODELS]


PROMPTING_METHODS = [
    "zero-shot-cot"
]

"""
Useful functions

"""

# helper for parsing input arguments using argparse library
def parse_input_args():
    parser = argparse.ArgumentParser(description="dual-process reasoning in LLMs")

    # adding arguments
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=DATASETS.keys())
    parser.add_argument("--model", type=str, default="gpt3", choices=SUPPORTED_MODELS)
    parser.add_argument("--prompting_method", type=str, default="zero-shot-cot", choices=PROMPTING_METHODS)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    return args

# helper for parsing input arguments for running experiments
def parse_experiment_args():
    exp_parser = argparse.ArgumentParser(description="dual-process reasoning in LLMs -> running experiments")
    exp_parser.add_argument("--dataset", type=str, default="gsm8k", choices=DATASETS.keys())
    exp_parser.add_argument("--model", type=str, default="gpt3", choices=SUPPORTED_MODELS)
    exp_parser.add_argument("--prompting_methods", type=str, default="zero-shot-cot", choices=PROMPTING_METHODS, nargs="+")
    exp_parser.add_argument("--num_samples", type=int, default=10)
    exp_parser.add_argument("--seed", type=int, default=42)

    args = exp_parser.parse_args()
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
    perplexity: float
    total_samples: int
    dropped_samples: int
    model_name: str
    dataset_name: str
    seed: int