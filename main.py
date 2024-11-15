"""
Language Models are Dual-Process Reasoners

Angel Patricio, Eva Ge, and Walta Teklezgi

MIT

2024
"""

# imports
import torch

from utils import *

if __name__ == "__main__":
    # get args from terminal
    args = parse_input_args()
    print("*************************************************")
    print(args)
    print("*************************************************\n")

    # figure out what device we're running on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}\n")

    # load dataset
    print("Loading data...\n")
    train, test = safe_load_data(args)
    print("Data loaded.\n")

    print(f"{train=}\n")
    print(f"{test=}\n")

    decoder = Decoder(args)

    print(f"Response from {args.model}:\n{decoder.generate_answer(args.prompt)}")

