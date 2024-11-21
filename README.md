# Language Models are Dual-Process Reasoners

## Setup

Starting from the root directory:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py <--kwargs>
```

## Args:

--dataset (str="gsm8k") : dataset name. one of ["gsm8k"]

--model (str="api") : model to use automatically set up for using the OpenAI API. one of ["gpt3",]

--prompting_method (str="zero-shot-cot") : prompting method to use, one of ["default", "zero-shot-cot", "dual-process"]

--num_samples (int=10) : number of samples to take from the given dataset (shuffled)

--seed (int=42) : random seed for reproducibility
