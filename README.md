# Language Models are Dual-Process Reasoners

## Setup

Starting from the root directory:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

huggingface-cli login

python main.py <--kwargs>
```

## Args:
note: everything in caps can be accessed via utils.py

--dataset (str="gsm8k") : dataset name. one of DATASETS

--model (str="gpt3") : model to use automatically set up for using the OpenAI API. one of MODELS

--prompting_method (str="zero-shot-cot") : prompting method to use, one of PROMPTING_METHODS

--num_samples (int=10) : number of samples to take from the given dataset (automatically shuffled)

--seed (int=42) : random seed for reproducibility
