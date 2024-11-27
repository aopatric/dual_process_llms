"""
Class for performing experiments with api calls slash local models
"""
import re
import ollama

from tqdm import tqdm

from utils import *
from prompting_examples import create_prefix

class Decoder:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.dset_info = DATASETS[self.args.dataset]
        self.model = args.model
        self.num_params = "70B" if self.model == "llama3.1:70b" else "8B"
        print(f"Setting up local instance of Llama 3.1 {self.num_params}...\n")
   
    def transform_prompt(self, base_prompt: str) -> str:
        # grab the method we passed into args
        method = self.args.prompting_method
        prefix = create_prefix(method)

        # process the prompt based on the method
        if method == "default":
            return base_prompt
        elif method == 'zero-shot-cot':
            return "Q: " + base_prompt + "\nA: " + prefix
        elif method in ["few-shot-cot", "dual-process", "dual-process-w-err"]:
            return prefix + "\n\nQ: " + base_prompt + "\nA: "
        else:
            return "other method detected!"

    def generate_answer(self, prompt: str):
        model_response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False
        ).response
        return model_response

    def get_final_ans(self, raw_ans):
        trigger = self.dset_info["answer_trigger"]

        if trigger:
            if trigger in raw_ans:
                answer = raw_ans.split(trigger)[-1].strip()

                try:
                    return float(answer)
                except ValueError:
                    # TODO: fix this so that it doesn't die when you try to use the MATH dataset
                    print(f"couldn't get value from answer '{answer}'\n")
            else:
                print(f"Trigger not found in example...\n")

        # Fallback to the last number if no trigger or not found in the answer
        numbers = re.findall(r'-?\d*\.?\d+', raw_ans)
        return float(numbers[-1]) if numbers else None
    
    # def get_perplexity(self, prompt: str, answer: str):
    #     if self.model_info["type"] == "api": # api models don't let us directly calculate perplexity
    #         raise ValueError("Perplexity not supported for API models")
        
    #     else:
    #         full_seq = prompt + " " + answer
    #         inputs = self.tokenizer(full_seq, return_tensors="pt").to(self.device)

    #         with torch.no_grad():
    #             outputs = self.model(**inputs)
    #             log_probs = outputs.logits

    #         # perplexity is the exponential of negative average log likelihood
    #         return torch.exp(-log_probs.mean()).item()
            


    def run_experiment(self):
        print(f"Running experiment with {self.args.num_samples} samples on dataset '{self.dset_info["path"]}' using prompting method {self.args.prompting_method}...")

        # extract final answers from dataset
        q_field, a_field = self.dset_info["question_field"], self.dset_info["answer_field"]
        questions, answers = self.data[q_field], self.data[a_field]
        true_final_ans = [self.get_final_ans(answer) for answer in answers]

        dropped = len([i for i in true_final_ans if i is None])

        correct = 0       

        # ppl_total = 0 if self.model_info["type"] == "local" else None

        # iterate over samples and test
        for i in tqdm(range(len(questions))):
            # get this data point
            q, a = questions[i], true_final_ans[i]

            prompt = self.transform_prompt(q)

            response = self.generate_answer(prompt)
            raw_ans = self.get_final_ans(response)

            # commented out because it's not supported for local models yet
            # if self.model_info["type"] == "local":
                # ppl_total += self.get_perplexity(prompt, response)

            print(f"True answer: {a}, Model answer: {raw_ans}")

            if a == raw_ans:
                correct += 1
        
        # get metrics to report
        final_acc = correct / (self.args.num_samples - dropped)
        # final_ppl = ppl_total / (self.args.num_samples - dropped) if ppl_total is not None else None
        print(f"Final accuracy: {final_acc}")

        return ExperimentResult(
            prompting_method=self.args.prompting_method,
            accuracy=final_acc,
            perplexity=None,
            total_samples=self.args.num_samples,
            dropped_samples=dropped,
            model_name=self.model,
            dataset_name=self.dset_info["path"],
            seed=self.args.seed
        )