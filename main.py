"""
Language Models are Dual-Process Reasoners

Angel Patricio, Eva Ge, and Walta Teklezgi

MIT

2024
"""

from utils import *

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
    decoder.run_experiment()



if __name__ == "__main__":
    main()