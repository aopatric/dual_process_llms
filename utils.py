"""
    Useful functions that make life easier for the rest of the code.
"""

"""
Imports

"""
import argparse
import os
import openai

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

"""
Constants

"""

SUPPORTED_DATASETS = [
    "gsm8k",
]

SUPPORTED_MODELS = [
    "gpt3",
]

API_MODELS = {
    "gpt3": "babbage-002",
    "davinci": "davinci-002"
}

PROMPTING_METHODS = [
    "default",
    "zero-shot-cot"
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
    parser.add_argument("--model", type=str, default="gpt3", choices=SUPPORTED_MODELS)
    parser.add_argument("--prompt", type=str, default="The color of the sky is ")
    parser.add_argument("--prompting_method", type=str, default="default", choices=PROMPTING_METHODS)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args

# safely load data by only the name, returns all splits of main branch
def safe_load_data(args):
    dataset = args.dataset
    seed = args.seed

    if dataset in CONFIG_MAP:
        train = load_dataset(dataset, CONFIG_MAP[dataset], split="train").shuffle(seed=seed)
        test = load_dataset(dataset, CONFIG_MAP[dataset], split="test").shuffle(seed=seed)
    else:
        train = load_dataset(dataset, split="train").shuffle(seed=seed)
        test = load_dataset(dataset, split="test").shuffle(seed=seed)

    return (train, test)

class Decoder:
    def __init__(self, args):
        self.args = args
        model_name = self.args.model

        # make sure we have a key if we want to use an api model
        if model_name in API_MODELS:
            assert os.getenv("OPENAI_API_KEY") is not None, "API key required for API model !"

            self.client = openai.OpenAI()

        # otherwise, set up the local model that will be ran instead
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            accelerator = Accelerator()

            self.model, self.tokenizer = accelerator.prepare(model, tokenizer)
    
    def transform_prompt(self, base_prompt: str) -> str:
        # grab the method we passed into args
        method = self.args.prompting_method

        # process the prompt based on the method
        if method == "default":
            return base_prompt
        elif method == "zero-shot-cot":
            return "Got to zero shot!!!"
        else:
            return "other method detected!"

    def generate_answer(self, base_prompt: str):
        model = self.args.model

        # turn the raw prompt into the modified one for testing
        prompt : str = self.transform_prompt(base_prompt)

        # print that prompt before running inference
        print(f"{prompt=}\n")

        # if using api, run the inference and store the raw result
        if model in API_MODELS:
            answer = openai.completions.create(
                model=API_MODELS[model],
                prompt=prompt
            ).choices[0].text
        else:
            # TODO: add support for the local inference version
            raise NotImplementedError

        # if using local model, run the inference locally and store the result

        # borrow code from zero shot cot to cleanse the answers and return cleaned answer
        return answer
