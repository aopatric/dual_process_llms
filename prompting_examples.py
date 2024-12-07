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

DUAL_PROCESS_COT = [
    "Q: If a train travels 60 miles per hour for 3 hours, how far does it travel?\nSystem 1: The train is moving fast, covering a lot of ground quickly.\nSystem 2: Calculate the distance using the formula Distance = Speed × Time. Distance = 60 miles/hour × 3 hours = 180 miles.\nTherefore: The train travels 180 miles.\nThe answer is 180.",
    
    "Q: Sarah has 3 times as many apples as Tom. Tom has 4 apples. How many apples do they have together?\nSystem 1: Sarah has a lot more apples than Tom.\nSystem 2: Calculate Sarah's apples: 3 × 4 = 12. Total apples = Sarah's apples + Tom's apples = 12 + 4 = 16.\nTherefore: Together, they have 16 apples.\nThe answer is 16.",
    
    "Q: A rectangle has a length of 10 cm and a width of 5 cm. What is the area of the rectangle?\nSystem 1: The rectangle is not very large.\nSystem 2: Calculate the area using the formula Area = Length × Width. Area = 10 cm × 5 cm = 50 cm².\nTherefore: The area of the rectangle is 50 square centimeters.\nThe answer is 50.",
    
    "Q: A bag contains 3 red balls and 2 blue balls. What is the probability of drawing a red ball?\nSystem 1: There are more red balls than blue balls.\nSystem 2: Calculate the probability: Number of red balls / Total number of balls = 3/5.\nTherefore: The probability of drawing a red ball is 3/5.\nThe answer is 0.6.",
    
    "Q: John buys 3 books for $12 each and 2 notebooks for $5 each. How much money does he spend in total?\nSystem 1: John is spending a fair amount of money.\nSystem 2: Calculate the total cost: Cost of books = 3 × $12 = $36. Cost of notebooks = 2 × $5 = $10. Total cost = $36 + $10 = $46.\nTherefore: John spends $46 in total.\nThe answer is 46.",
    
    "Q: A store has a 30% off sale. If a shirt originally costs $40, how much will it cost after the discount?\nSystem 1: The discount is significant, reducing the price considerably.\nSystem 2: Calculate the final price: 30% off means paying 70% of the original price. Final price = $40 × 0.7 = $28.\nTherefore: The shirt will cost $28 after the discount.\nThe answer is 28.",
    
    "Q: If it takes 8 hours to paint 3 rooms, how many hours will it take to paint 7 rooms at the same rate?\nSystem 1: Painting more rooms will take significantly more time.\nSystem 2: Calculate the time per room: 8 hours ÷ 3 rooms = 2.67 hours per room. Time for 7 rooms = 2.67 × 7 = 18.67 hours.\nTherefore: It will take 18.67 hours to paint 7 rooms.\nThe answer is 18.67.",
    
    "Q: A car uses 6 gallons of gas to drive 180 miles. How many miles per gallon does the car get?\nSystem 1: The car seems to be quite efficient.\nSystem 2: Calculate miles per gallon: Total distance ÷ Total gas = 180 miles ÷ 6 gallons = 30 miles per gallon.\nTherefore: The car gets 30 miles per gallon.\nThe answer is 30.",
    
    "Q: If 4 pencils cost $6, how much would 10 pencils cost?\nSystem 1: More pencils will cost proportionally more.\nSystem 2: Calculate the cost per pencil: $6 ÷ 4 = $1.50 per pencil. Cost for 10 pencils = $1.50 × 10 = $15.\nTherefore: 10 pencils would cost $15.\nThe answer is 15.",
    
    "Q: A recipe calls for 2.5 cups of flour to make 20 cookies. How many cups of flour are needed to make 50 cookies?\nSystem 1: More cookies require more flour.\nSystem 2: Calculate cups per cookie: 2.5 cups ÷ 20 cookies = 0.125 cups per cookie. Cups for 50 cookies = 0.125 × 50 = 6.25 cups.\nTherefore: 6.25 cups of flour are needed.\nThe answer is 6.25."
]

DUAL_PROCESS_W_ERR_COT = [
    """Q: If you have 5 boxes with 6 chocolates in each box, how many chocolates do you have total?
System 1: This feels like it should be around 30 chocolates.
System 2: Let's calculate: 5 boxes × 6 chocolates = 30 chocolates.
System 1 and System 2 agree on this calculation.
Therefore: There are 30 chocolates in total.
The answer is 30.""",

    """Q: What is 25% of 200?
System 1: A quarter of 200 should be 50.
System 2: Let's calculate: 200 × 0.25 = 50.
System 1 and System 2 agree perfectly here.
Therefore: 25% of 200 is 50.
The answer is 50.""",

    """Q: If a square has sides of length 10 meters, what is its area?
System 1: With sides of 10, the area should be 100 square meters.
System 2: Let's calculate: Area = side × side = 10 × 10 = 100 square meters.
System 1 and System 2 are in perfect agreement.
Therefore: The square's area is 100 square meters.
The answer is 100.""",

    """Q: What is double 45?
System 1: Doubling 45 should give us 90.
System 2: Let's calculate: 45 × 2 = 90.
System 1 and System 2 agree on this straightforward calculation.
Therefore: Double 45 is 90.
The answer is 90.""",

    """Q: A rectangle has a length of 8 cm and a width of 3 cm. What is the perimeter?
System 1: The perimeter should be the sum of all sides, which feels like 22 cm.
System 2: Let's calculate: Perimeter = 2 × (Length + Width) = 2 × (8 + 3) = 22 cm.
System 1 and System 2 agree on this calculation.
Therefore: The perimeter is 22 cm.
The answer is 22.""",

    """Q: If a car travels at 60 miles per hour for 2 hours, how far does it travel?
System 1: It should travel about 120 miles.
System 2: Let's calculate: Distance = Speed × Time = 60 miles/hour × 2 hours = 120 miles.
System 1 and System 2 agree on this calculation.
Therefore: The car travels 120 miles.
The answer is 120.""",

    """Q: What is the sum of 15 and 25?
System 1: Adding 15 and 25 should give us 40.
System 2: Let's calculate: 15 + 25 = 40.
System 1 and System 2 agree on this calculation.
Therefore: The sum is 40.
The answer is 40.""",

    """Q: If a triangle has sides of 3 cm, 4 cm, and 5 cm, what is its perimeter?
System 1: The perimeter should be the sum of all sides, which feels like 12 cm.
System 2: Let's calculate: Perimeter = 3 + 4 + 5 = 12 cm.
System 1 and System 2 agree on this calculation.
Therefore: The perimeter is 12 cm.
The answer is 12.""",

    """Q: What is 10% of 150?
System 1: 10% of 150 should be 15.
System 2: Let's calculate: 150 × 0.10 = 15.
System 1 and System 2 agree on this calculation.
Therefore: 10% of 150 is 15.
The answer is 15.""",

    """Q: If you have 3 dozen eggs, how many eggs do you have?
System 1: 3 dozen should be 36 eggs.
System 2: Let's calculate: 3 × 12 = 36.
System 1 and System 2 agree on this calculation.
Therefore: You have 36 eggs.
The answer is 36.""",

    """Q: A store offers a 20% discount on a $80 jacket. What is the final price?
System 1: The price should be a bit more than half of $80.
System 2: Let me calculate: 20% of $80 is $80 × 0.2 = $16, so the final price is $16.
Wait - System 1 notices this seems too low. Let's check System 2's work:
The 20% ($16) is the discount amount, so we need to subtract it from the original price:
$80 - $16 = $64
Therefore: The jacket costs $64 after discount.
The answer is 64.""",

    """Q: If it takes 5 workers 6 hours to build a wall, how long would it take 15 workers?
System 1: With triple the workers, it should take about a third of the time.
System 2: Let's calculate: 15 workers is 3 times more workers, so it will take 6 × 3 = 18 hours.
Wait - System 1 notices this is wrong. More workers should make it faster, not slower. Let's correct System 2:
If 5 workers take 6 hours, then 15 workers (3 times more) will take 6 ÷ 3 = 2 hours.
Therefore: It will take 2 hours with 15 workers.
The answer is 2.""",

    """Q: A car travels 240 miles using 8 gallons of gas. What is its fuel efficiency in miles per gallon?
System 1: The efficiency should be pretty good, probably around 30 mpg.
System 2: Let's calculate: 8 gallons ÷ 240 miles = 0.033 gallons per mile.
Wait - System 1 notices this seems very low. System 2 divided the wrong way:
Correct calculation: 240 miles ÷ 8 gallons = 30 miles per gallon.
Therefore: The car's fuel efficiency is 30 miles per gallon.
The answer is 30.""",

    """Q: If 4 pizzas feed 12 people, how many pizzas are needed for 20 people?
System 1: It should be a bit more than 6 pizzas.
System 2: Let's set up a proportion: 4/12 = x/20, so x = (4 × 20) ÷ 12 = 6.67
Wait - System 1 confirms this matches our intuition of "a bit more than 6."
Therefore: We need 6.67 pizzas, which means ordering 7 pizzas to have enough.
The answer is 7.""",

    """Q: A shirt's price is increased by 50% and then decreased by 50%. What is the final price compared to the original?
System 1: It feels like it should return to the original price, but something seems off.
System 2: Let's say it starts at $100. After 50% increase: $100 + $50 = $150.
After 50% decrease: $150 - $50 = $100.
Wait - System 1 is right that something's wrong. System 2 made a mistake:
The second 50% should be calculated from $150, not $100:
50% of $150 = $75, so $150 - $75 = $75
Therefore: The final price is 75% of the original.
The answer is 75."""
]

DUAL_PROMPTING = DUAL_PROCESS_COT + DUAL_PROCESS_W_ERR_COT

def create_prefix(prompting_method, samples=8):
    if prompting_method == "chain-of-thought":
        selected = random.sample(CHAIN_OF_THOUGHT, min(samples, len(CHAIN_OF_THOUGHT)))
        return "\n\n".join(selected)
    elif prompting_method == "dual-prompting":
        selected = random.sample(DUAL_PROMPTING, min(samples, len(DUAL_PROMPTING)))
        return "\n\n".join(selected)
    else:
        return "Method not recognized!"