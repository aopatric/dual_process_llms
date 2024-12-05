"""
Class for performing experiments with api calls slash local models
"""
import re
import ollama

from tqdm import tqdm

from utils import DATASETS, ExperimentResult, PROMPTING_METHODS
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
        prefix = create_prefix(method, samples=self.args.n_shots)

        # process the prompt based on the method
        if method in PROMPTING_METHODS:
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
                print("Trigger not found in example...\n")

        # Fallback to the last number if no trigger or not found in the answer
        numbers = re.findall(r'-?\d*\.?\d+', raw_ans)
        return float(numbers[-1]) if numbers else None
   
    def run_experiment(self):
        # extract final answers from dataset
        q_field, a_field = self.dset_info["question_field"], self.dset_info["answer_field"]
        questions, answers = self.data[q_field], self.data[a_field]
        true_final_ans = [self.get_final_ans(answer) for answer in answers]

        dropped = len([i for i in true_final_ans if i is None])

        correct = 0       

        # iterate over samples and test
        for i in tqdm(range(len(questions))):
            # get this data point
            q, a = questions[i], true_final_ans[i]

            prompt = self.transform_prompt(q)

            response = self.generate_answer(prompt)
            raw_ans = self.get_final_ans(response)
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
            total_samples=self.args.num_samples,
            dropped_samples=dropped,
            model_name=self.model,
            dataset_name=self.dset_info["path"],
            seed=self.args.seed,
            n_shots=self.args.n_shots
        )