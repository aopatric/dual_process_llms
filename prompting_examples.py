"""

Compilation of all of the prompting examples that we need for generating few-shot prompts

"""

import random


ZERO_SHOT_COT = [
    "Let's solve this step by step:",
    "Let's approach this systematically:",
    "Let's break this down step by step:",
    "Let's put our thinking caps on:",
]

FEW_SHOT_COT = [
   "Q: If a train travels 60 miles per hour for 3 hours, how far does it travel?\nA: Let's solve this step by step:\n1. The train travels at 60 miles per hour\n2. It travels for 3 hours\n3. Distance = Speed × Time = 60 miles/hour × 3 hours = 180 miles\nTherefore, the train travels 180 miles.\n\nThe answer is 180.",
   
   "Q: Sarah has 3 times as many apples as Tom. Tom has 4 apples. How many apples do they have together?\nA: Let's solve this step by step:\n1. Tom has 4 apples\n2. Sarah has 3 times as many apples as Tom: 3 × 4 = 12 apples\n3. Total apples = Sarah's apples + Tom's apples = 12 + 4 = 16 apples\nTherefore, they have 16 apples together.\n\nThe answer is 16.",
   
   "Q: A rectangle has a length of 10 cm and a width of 5 cm. What is the area of the rectangle?\nA: Let's solve this step by step:\n1. Length of the rectangle = 10 cm\n2. Width of the rectangle = 5 cm\n3. Area = Length × Width = 10 cm × 5 cm = 50 cm²\nTherefore, the area of the rectangle is 50 square centimeters.\n\nThe answer is 50.",
   
   "Q: A bag contains 3 red balls and 2 blue balls. What is the probability of drawing a red ball?\nA: Let's solve this step by step:\n1. Total number of balls = 3 red + 2 blue = 5 balls\n2. Number of red balls = 3\n3. Probability = Number of red balls / Total number of balls = 3/5\nTherefore, the probability of drawing a red ball is 3/5.\n\nThe answer is 0.6.",
   
   "Q: John buys 3 books for $12 each and 2 notebooks for $5 each. How much money does he spend in total?\nA: Let's solve this step by step:\n1. Cost of books = 3 × $12 = $36\n2. Cost of notebooks = 2 × $5 = $10\n3. Total cost = Cost of books + Cost of notebooks = $36 + $10 = $46\nTherefore, John spends $46 in total.\n\nThe answer is 46.",

   "Q: A store has a 30% off sale. If a shirt originally costs $40, how much will it cost after the discount?\nA: Let's solve this step by step:\n1. Original price is $40\n2. 30% off means we pay 70% of the original price\n3. 70% as a decimal is 0.7\n4. Final price = $40 × 0.7 = $28\nTherefore, the shirt will cost $28 after the discount.\n\nThe answer is 28.",
   
   "Q: If it takes 8 hours to paint 3 rooms, how many hours will it take to paint 7 rooms at the same rate?\nA: Let's solve this step by step:\n1. Time for 3 rooms = 8 hours\n2. Time for 1 room = 8 ÷ 3 = 2.67 hours\n3. Time for 7 rooms = 2.67 × 7 = 18.67 hours\nTherefore, it will take 18.67 hours to paint 7 rooms.\n\nThe answer is 18.67.",
   
   "Q: A car uses 6 gallons of gas to drive 180 miles. How many miles per gallon does the car get?\nA: Let's solve this step by step:\n1. Total distance = 180 miles\n2. Total gas used = 6 gallons\n3. Miles per gallon = Total distance ÷ Total gas\n4. Miles per gallon = 180 ÷ 6 = 30\nTherefore, the car gets 30 miles per gallon.\n\nThe answer is 30.",
   
   "Q: If 4 pencils cost $6, how much would 10 pencils cost?\nA: Let's solve this step by step:\n1. Cost of 4 pencils = $6\n2. Cost of 1 pencil = $6 ÷ 4 = $1.50\n3. Cost of 10 pencils = $1.50 × 10 = $15\nTherefore, 10 pencils would cost $15.\n\nThe answer is 15.",
   
   "Q: A recipe calls for 2.5 cups of flour to make 20 cookies. How many cups of flour are needed to make 50 cookies?\nA: Let's solve this step by step:\n1. 2.5 cups makes 20 cookies\n2. Cups needed for 1 cookie = 2.5 ÷ 20 = 0.125 cups\n3. Cups needed for 50 cookies = 0.125 × 50 = 6.25 cups\nTherefore, 6.25 cups of flour are needed.\n\nThe answer is 6.25.",
   
   "Q: In a class of 30 students, 40% are boys. How many girls are in the class?\nA: Let's solve this step by step:\n1. Total students = 30\n2. Percentage of boys = 40%\n3. Number of boys = 30 × 0.4 = 12\n4. Number of girls = Total - Boys = 30 - 12 = 18\nTherefore, there are 18 girls in the class.\n\nThe answer is 18.",
   
   "Q: A train leaves at 2:30 PM and arrives at 5:45 PM. How many minutes was the journey?\nA: Let's solve this step by step:\n1. Start time = 2:30 PM\n2. End time = 5:45 PM\n3. Hours difference = 3 hours and 15 minutes\n4. Convert to minutes: (3 × 60) + 15 = 195 minutes\nTherefore, the journey was 195 minutes.\n\nThe answer is 195.",
   
   "Q: If you have $100 and spend 15% on food and 25% on rent, how much money do you have left?\nA: Let's solve this step by step:\n1. Total money = $100\n2. Spent on food = $100 × 0.15 = $15\n3. Spent on rent = $100 × 0.25 = $25\n4. Total spent = $15 + $25 = $40\n5. Money left = $100 - $40 = $60\nTherefore, you have $60 left.\n\nThe answer is 60."
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