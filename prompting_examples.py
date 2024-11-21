"""

Compilation of all of the prompting examples that we need for generating few-shot prompts

"""

import random


ZERO_SHOT_COT = [
    "Let's solve this step by step:",
    "Let's approach this systematically:",
    "Let's break this down step by step:"
]

FEW_SHOT_COT = [

]

DUAL_PROCESS_COT = [

]

METHODS_TO_PROMPTS = {
    "zero-shot-cot": ZERO_SHOT_COT,
    "few-shot-cot": FEW_SHOT_COT,
    "dual-process": DUAL_PROCESS_COT
}

def create_prefix(prompting_method, samples=3):
    options = METHODS_TO_PROMPTS[prompting_method]

    selected = random.sample(options, min(samples, len(options)))

    # only need the prefix to the answer for zero-shot-cot
    if prompting_method == "zero-shot-cot":
        return selected[0]

    # othwise, join the selected examples and return the prefix
    return "\n\n".join(selected)