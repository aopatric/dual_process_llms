"""

Compilation of all of the prompting examples that we need for generating few-shot prompts

"""

import random

CHAIN_OF_THOUGHT = [
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


DUAL_PROMPTING = [
    """Q: Suppose there are 6 identical chairs, and each chair has 4 legs. How many chair legs are there in total?

System 1: Intuitively, 6 chairs times 4 legs per chair should give me 24 legs. That seems straightforward.

System 2: Let's be exact: 6 chairs * 4 legs/chair = 24 legs total. There's no trick here, just multiplication.

System 1: My initial guess matches the careful calculation perfectly.

System 2: Great, no contradictions. The final answer is 24.

Final Answer: 24""",

    """Q: You have 10 notebooks, and each notebook contains 50 pages. After tearing out 8 pages from one notebook, how many pages remain in total?

System 1: Let’s think it through: Initially, 10 notebooks * 50 pages each = 500 pages total. If I remove 8 pages from one notebook, I lose those 8 pages from the total. So 500 - 8 = 492 pages remain. That feels right.

System 2: Carefully checking: 10 * 50 = 500 pages initially. Removing 8 pages from the total leaves 500 - 8 = 492 pages. Straightforward arithmetic, no complexity missed.

System 1: My gut calculation and the careful method match up perfectly.

System 2: Exactly. There's no ambiguity. The total remaining is 492.

Final Answer: 492""",

    """Q: You have 3 baskets of eggs. Each basket contains 12 eggs. You want to give 5 eggs away. How many eggs remain?

System 1: Total eggs: 3 baskets * 12 eggs = 36 eggs. If I give away 5, then 36 - 5 = 31 eggs remain. That seems correct.

System 2: Double-check: Total: 36 eggs. Remove 5: 36 - 5 = 31 eggs. Straight math, no hidden catch.

System 1: My initial intuition and the calculated result align perfectly.

System 2: Perfect. The final answer is 31.

Final Answer: 31""",

    """Q: You plan a hike of 15 miles. If you walk at a steady pace of 3 miles per hour, how many hours will it take to complete the hike?

System 1: Rough intuition: 15 miles divided by 3 mph = 5 hours. That seems right.

System 2: Precise calculation: 15 miles ÷ 3 mph = 5 hours. Exactly as the intuition stated.

System 1: Matches perfectly, no second thoughts.

System 2: Straightforward. The answer is 5 hours.

Final Answer: 5""",

    """Q: You have a room with 4 walls, and each wall requires 2 cans of paint. If you paint all 4 walls, how many cans of paint do you need in total?

System 1: Quickly, 4 walls * 2 cans each = 8 cans total. Feels right.

System 2: Check carefully: 4 walls × 2 cans/wall = 8 cans total. No trick, just multiplication.

System 1: Perfect agreement.

System 2: The answer is definitively 8.

Final Answer: 8""",

    """Q: You have 100 items to pack, and each box can hold 8 items. How many boxes do you need if you want no items left unpacked?

System 1: Let’s estimate: 100 ÷ 8 = 12.5, so I’d need 13 boxes because you can't have half a box. I’m guessing 13 boxes total.

System 2: Exact division: 100 ÷ 8 = 12 remainder 4. Since there's a remainder, I need one more box. That’s 12 full boxes plus 1 extra = 13 boxes total.

System 1: Matches the careful reasoning perfectly.

System 2: Correct. The final answer is 13 boxes.

Final Answer: 13""",

    """Q: You drive 150 miles in 3 hours. What is your average speed in miles per hour?

System 1: Intuition: 150 miles / 3 hours = 50 mph average. Straightforward.

System 2: Step-by-step: Average speed = Total distance ÷ Total time = 150 ÷ 3 = 50 mph. No complexity.

System 1: Perfect match again.

System 2: Final answer is 50 mph.

Final Answer: 50""",

    """Q: You need to cover a floor that measures 10 feet by 12 feet with square tiles that are 1 foot by 1 foot. How many tiles do you need in total?

System 1: The floor area is 10 * 12 = 120 square feet. Each tile covers 1 square foot. So I need 120 tiles total. Seems right.

System 2: Careful reasoning: The floor area = 10 ft * 12 ft = 120 sq ft. Each 1x1 tile covers 1 sq ft. Thus, total tiles needed = 120.

System 1: Matches perfectly.

System 2: Exactly. The final answer is 120.

Final Answer: 120""",

    """Q: You have 2 pizzas, and each pizza is cut into 8 slices. If you have 5 friends and want to give each friend 3 slices, will you have enough slices?

System 1: Total slices: 2 pizzas * 8 slices each = 16 slices. You need 5 friends * 3 slices/friend = 15 slices. 16 slices available vs. 15 needed. Yes, I have enough slices, and even 1 slice leftover.

System 2: Double-check: 2*8 = 16 slices total. 5*3 = 15 slices required. 16 ≥ 15, so yes, there's enough.

System 1: Perfect match. I have one slice more than needed.

System 2: Answer: Yes, I have enough, with 1 slice extra.

Final Answer: Yes, enough slices (1 slice remaining).""",

    """Q: An orchard has 10 apple trees, and each tree grows 25 apples. If you pick 50 apples to give away, how many apples are left on the trees?

System 1: Total apples: 10 trees * 25 apples/tree = 250 apples. After giving away 50, 250 - 50 = 200 apples remain. Seems correct.

System 2: Careful check: 10 * 25 = 250. Remove 50: 250 - 50 = 200. Straight arithmetic.

System 1: Matches exactly.

System 2: The final answer is 200 apples.

Final Answer: 200""",

    """Q: You are putting up a fence along a straight line. For every 10 feet of fence, you place 1 post. If the fence line is 100 feet long, how many posts do you need in total?

System 1: Intuitively, if it's every 10 feet, 100 feet would have 100/10 = 10 posts. But wait, fences often require one extra post at the end. Let’s think carefully: If you place a post at 0 ft, then at 10 ft, 20 ft, ... up to 100 ft, that's actually 11 posts (one at the start, one at every increment of 10, and one at the end).

System 2: Let's reason systematically. If you have a line of length L, and posts every X feet, you count the endpoints. For a 100-ft line at intervals of 10 ft, the posts are at: 0,10,20,30,40,50,60,70,80,90,100. That's indeed 11 posts total.

System 1: Good thing I double-checked. It's 11, not 10.

System 2: Perfect reasoning. The final answer is 11 posts.

Final Answer: 11""",

    """Q: You have a fundraiser where you must give each of 20 participants 3 chocolate bars. Chocolate bars are sold in boxes of 5 bars. How many boxes must you buy to have enough for everyone?

System 1: Total needed: 20 participants * 3 bars each = 60 bars. Each box has 5 bars, so 60 ÷ 5 = 12 boxes. That's a clean division, no remainder.

System 2: Carefully verify: Need 60 bars total. Each box has 5 bars. 60 ÷ 5 = 12 exactly, so 12 boxes total.

System 1: Perfect. No discrepancy.

System 2: The final answer is 12 boxes.

Final Answer: 12"""
]

def create_prefix(prompting_method, samples=8):
    if prompting_method == "chain-of-thought":
        selected = random.sample(CHAIN_OF_THOUGHT, min(samples, len(CHAIN_OF_THOUGHT)))
        return "\n\n".join(selected)
    elif prompting_method == "dual-prompting":
        selected = random.sample(DUAL_PROMPTING, min(samples, len(DUAL_PROMPTING)))
        return "\n\n".join(selected)
    else:
        return "Method not recognized!"