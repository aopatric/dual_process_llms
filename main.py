"""
Language Models are Dual-Process Reasoners

Angel Patricio, Eva Ge, and Walta Teklezgi

MIT

2024
"""

# imports
import transformers
import accelerate
import torch

from utils import *

if __name__ == "__main__":
    # get args from terminal
    args = parse_input_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device}")

    print("Loading data...")
    train, test = safe_load_data(args.dataset)
    print("Data loaded.")
