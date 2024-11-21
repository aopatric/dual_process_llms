"""
    Useful functions that make life easier for the rest of the code.
"""

"""
Imports

"""
import argparse
import os
import openai
import re

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm

"""
Constants

"""

DATASETS = {
    "gsm8k" : {
        "answer_trigger": "#### ", # gsm8k final answers preceded by this prefix
        "path": "gsm8k",
        "branch": "main",
        "answer_field": "answer"
    }
}

SUPPORTED_MODELS = [
    "gpt3",
    "davinci",
    "gpt4"
]

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
    "o1mini": {
        "type": "api",
        "name": "o1-mini-2024-09-12",
        "api_type": "chat"
    }
}


PROMPTING_METHODS = [
    "default",
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
    parser.add_argument("--prompt", type=str, default="The color of the sky is ")
    parser.add_argument("--prompting_method", type=str, default="zero-shot-cot", choices=PROMPTING_METHODS)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    return args

# safely load data by the arguments
def safe_load_data(args):
    config = DATASETS[args.dataset]
    n_samples = args.num_samples
    seed = args.seed

    # get a random subset of length n_samples from the requested dataset
    data = load_dataset(
                        args.dataset,
                        config["branch"],
                        split="test"
                    ).shuffle(seed=seed)[:n_samples]
    return data



class Decoder:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.model_info = MODELS[self.args.model]

        # make sure we have a key if we want to use an api model
        if self.model_info["type"] == "api":
            assert os.getenv("OPENAI_API_KEY") is not None, "API key required for API model !"

            self.client = openai.OpenAI()

        # otherwise, set up the local model that will be ran instead
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info["name"])
            model = AutoModelForCausalLM.from_pretrained(self.model_info["name"])
            accelerator = Accelerator()

            self.model, self.tokenizer = accelerator.prepare(model, tokenizer)
    
    def transform_prompt(self, base_prompt: str) -> str:
        # grab the method we passed into args
        method = self.args.prompting_method

        # process the prompt based on the method
        if method == "default":
            return base_prompt
        elif method == "zero-shot-cot":
            return "Q: " + base_prompt + "\nA: Let's think step by step: "
        else:
            return "other method detected!"

    def generate_answer(self, prompt: str):
        model = self.args.model

        # if using api, run the inference and store the raw result
        if self.model_info["type"] == "api":
            # check if we have a completion or a chat model
            if self.model_info["api_type"] == "chat":
                answer = openai.chat.completions.create(
                    model=self.model_info["name"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.0
                ).choices[0].message.content
            else:
                answer = openai.completions.create(
                    model=self.model_info["name"],
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.0,
                    stop=["Q:", "\n\n"]
                ).choices[0].text
        else:
            # TODO: add support for the local inference version
            raise NotImplementedError

        # if using local model, run the inference locally and store the result

        # borrow code from zero shot cot to cleanse the answers and return cleaned answer
        return answer
    
    def get_final_ans(self, raw_ans):
        trigger = DATASETS[self.args.dataset]["answer_trigger"]

        if trigger in raw_ans:
            answer = raw_ans.split(trigger)[-1].strip()

            try:
                return float(answer)
            except ValueError:
                print(f"couldn't get value from answer '{raw_ans}'")
                return None 
        print(f"Trigger not found in answer '{raw_ans}' from dataset '{self.args.datset}'")
        return None

    def extract_answer(self, text: str) -> float:
        """Extract final numerical answer using dataset-specific trigger."""
        trigger = DATASETS[self.args.dataset]["answer_trigger"]
        
        if trigger in text:
            after_trigger = text.split(trigger)[-1].strip()
            try:
                return float(after_trigger)
            except ValueError:
                pass
            
        # Fallback to last number
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return float(numbers[-1]) if numbers else None

    def run_experiment(self):
        print(f"Running experiment with {self.args.num_samples} samples on dataset '{self.args.dataset}'...")

        # extract final answers from dataset
        questions, answers = self.data["question"], self.data[DATASETS[self.args.dataset]["answer_field"]]
        true_final_ans = [self.get_final_ans(answer) for answer in answers]

        correct = 0       

        # iterate over samples and test
        for i in tqdm(range(len(questions))):
            # get this data point
            q, a = questions[i], true_final_ans[i]

            prompt = self.transform_prompt(q)

            response = self.generate_answer(prompt)
            raw_ans = self.extract_answer(response)
            
            print(f"True answer: {a}, Model answer: {raw_ans}")

            if a == raw_ans:
                correct += 1
        
        print(f"\nFinal accuracy: Model {self.args.model} with prompting method {self.args.prompting_method} had an accuracy of {correct / self.args.num_samples} on {self.args.num_samples} from '{self.args.dataset}'")