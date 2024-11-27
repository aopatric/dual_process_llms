"""
Language Models are Dual-Process Reasoners

Angel Patricio, Eva Ge, and Walta Teklezgi

MIT

2024
"""
import os
import json

from utils import *
from decoder import Decoder
from dataclasses import asdict   
from datetime import datetime

def main():
    # get args from terminal
    args = parse_input_args()
    print("*************************************************")
    print(args)
    print("*************************************************\n")


    # load dataset (note this data is already shuffled, so grabbing the first n later is okay)
    print("Loading data...\n")
    data = safe_load_data(args)
    print("Data loaded.\n")

    decoder = Decoder(args, data)

    # run experiments
    result = decoder.run_experiment()

    # save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = os.path.join(args.output_dir, f"expresults_{args.dataset}_{args.prompting_method}_{timestamp}.json")
        with open(filename, "w") as f:
            json.dump(asdict(result), f, indent=2)

if __name__ == "__main__":
    main()