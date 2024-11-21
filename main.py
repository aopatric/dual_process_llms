"""
Language Models are Dual-Process Reasoners

Angel Patricio, Eva Ge, and Walta Teklezgi

MIT

2024
"""

# imports
import torch

from utils import *

def main():
    # get args from terminal
    args = parse_input_args()
    print("*************************************************")
    print(args)
    print("*************************************************\n")

    # figure out what device we're running on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Local model running on: {device}\n")

    # note that it's unused if using api
    model_type = MODELS[args.model]["type"]
    if model_type == "api":
        print(f"Model '{args.model}' only supports API inference. Local instance unused.")

    # load dataset (note this data is already shuffled, so grabbing the first n later is okay)
    print("Loading data...\n")
    data = safe_load_data(args)
    print("Data loaded.\n")

    decoder = Decoder(args, data)

    # run experiments
    decoder.run_experiment()



if __name__ == "__main__":
    main()