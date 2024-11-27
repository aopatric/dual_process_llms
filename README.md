# Language Models are Dual-Process Reasoners

## Usage

Starting from the root directory:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_experiments.py <--kwargs>
```

## Args:
note: everything in caps can be accessed via utils.py

--dataset (str="gsm8k") : dataset name. one of DATASETS (gsm8k, mathqa, math)

--prompting_methods (str="zero-shot-cot") : prompting methods to use, one of PROMPTING_METHODS. Can specify multiple methods.

--model (str="llama3.1:8b") : model to use. one of ["llama3.1:70b", "llama3.1:8b"]

--num_samples (int=10) : number of samples to take from the given dataset (automatically shuffled)

--seed (int=42) : random seed for reproducibility