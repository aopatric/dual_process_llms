"""
Class for performing experiments with api calls slash local models
"""
import os
import torch
import re
import openai

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import *
from prompting_examples import create_prefix

class Decoder:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.model_info = MODELS[self.args.model]
        self.dset_info = DATASETS[self.args.dataset]

        # make sure we have a key if we want to use an api model
        if self.model_info["type"] == "api":
            print("Connecting to OpenAI API...\n")
            assert os.getenv("OPENAI_API_KEY") is not None, "API key required for API model !"

            self.client = openai.OpenAI()

        # otherwise, set up the local model that will be ran instead
        elif self.model_info["type"] == "local":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Setting up local instance of {self.model_info["name"]} on {self.device}...\n")

            # quantizing models to 4 bits to support larger model inference
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_info["tokenizer"],
                token=HF_TOKEN
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_info["name"],
                device_map="auto",
                quantization_config=config,
                trust_remote_code=True,
                token=HF_TOKEN
            )

            print(f"Model '{self.model_info["name"]}' successfully loaded.\n")
    
    def transform_prompt(self, base_prompt: str) -> str:
        # grab the method we passed into args
        method = self.args.prompting_method
        prefix = create_prefix(method, samples=5)
        print(prefix)

        # process the prompt based on the method
        if method == "default":
            return base_prompt
        elif method == "zero-shot-cot":
            return "Q: " + base_prompt + "\nA: " + prefix
        else:
            return "other method detected!"

    def generate_answer(self, prompt: str):
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
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **tokens,
                max_new_tokens=256,
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def get_final_ans(self, raw_ans):
        trigger = self.dset_info["answer_trigger"]

        if trigger:
            if trigger in raw_ans:
                answer = raw_ans.split(trigger)[-1].strip()

                try:
                    return float(answer)
                except ValueError:
                    # TODO: fix this so that it doesn't die when you try to use the MATH dataset
                    print(f"couldn't get value from answer '{answer}'")
            else:
                print(f"Trigger not found in answer '{raw_ans}' from dataset '{self.dset_info["path"]}'")

        # Fallback to the last number if no trigger or not found in the answer
        numbers = re.findall(r'-?\d*\.?\d+', raw_ans)
        return float(numbers[-1]) if numbers else None
    
    def get_perplexity(self, prompt: str, answer: str):
        if self.model_info["type"] == "api": # api models don't let us directly calculate perplexity
            raise ValueError("Perplexity not supported for API models")
        
        else:
            full_seq = prompt + " " + answer
            inputs = self.tokenizer(full_seq, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                log_probs = outputs.logits

            # perplexity is the exponential of negative average log likelihood
            return torch.exp(-log_probs.mean()).item()
            


    def run_experiment(self):
        print(f"Running experiment with {self.args.num_samples} samples on dataset '{self.dset_info["path"]}' using prompting method {self.args.prompting_method}...")

        # extract final answers from dataset
        q_field, a_field = self.dset_info["question_field"], self.dset_info["answer_field"]
        questions, answers = self.data[q_field], self.data[a_field]
        true_final_ans = [self.get_final_ans(answer) for answer in answers]

        dropped = len([i for i in true_final_ans if i is None])

        correct = 0       

        ppl_total = 0 if self.model_info["type"] == "local" else None

        # iterate over samples and test
        for i in tqdm(range(len(questions))):
            # get this data point
            q, a = questions[i], true_final_ans[i]

            prompt = self.transform_prompt(q)

            response = self.generate_answer(prompt)
            raw_ans = self.get_final_ans(response)

            if self.model_info["type"] == "local":
                ppl_total += self.get_perplexity(prompt, response)

            print(f"True answer: {a}, Model answer: {raw_ans}")

            if a == raw_ans:
                correct += 1
        
        # get metrics to report
        final_acc = correct / (self.args.num_samples - dropped)
        final_ppl = ppl_total / (self.args.num_samples - dropped) if ppl_total is not None else None
        print(f"Final accuracy: {final_acc}, Final perplexity: {final_ppl}")

        return ExperimentResult(
            prompting_method=self.args.prompting_method,
            accuracy=final_acc,
            perplexity=final_ppl,
            total_samples=self.args.num_samples,
            dropped_samples=dropped,
            model_name=self.model_info["name"],
            dataset_name=self.dset_info["path"],
            seed=self.args.seed
        )