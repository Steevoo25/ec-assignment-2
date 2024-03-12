import argparse
import sexpdata as sex
import math
import numpy as np
import time
import random


# ------------
# CONSTANTS
# ------------
SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"
SAMPLE_EXP_3 = "(max (sub (mul 2 3) (add 1 1)) (exp (add 4 6)))"

FUNCTION_NODES = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4),('data', 1),('diff', 2),('avg', 2)]

# ------------
# QUESTION 1
# ------------

# adds 2 expressions
def add(exp1, exp2, n, x):
    return float(evaluate(exp1, n, x)) + float(evaluate(exp2, n, x))

# Subtracts exp2 from exp1
def sub (exp1, exp2, n, x):
    return float(evaluate(exp1, n, x)) - float(evaluate(exp2, n, x))
    
# Multiplies 2 expressions
def mul(exp1, exp2, n, x):
    return float(evaluate(exp1, n, x)) * float(evaluate(exp2, n, x))

# Divides exp1 by exp 2 as long as exp2 is not 0
def div(exp1, exp2, n, x):
    exp2_eval = float(evaluate(exp2, n, x))
    exp1_eval = float(evaluate(exp1, n, x))
    if exp2_eval == 0:
        return 0
    else:
        return exp1_eval / exp2_eval
        
# Returns exp1 to the power of exp2
def pow(exp1, exp2, n, x):
    return float(evaluate(exp1, n, x)) ** float(evaluate(exp2, n, x))

# Returns the square root of exp1
def sqrt(exp1, n, x):
    exp1_eval = float(evaluate(exp1, n, x))
    if exp1_eval < 0:
        return 0
    return math.sqrt(exp1_eval)

# Takes log2 of 1 expression
def log(exp1, n, x):
    exp1_val = float(evaluate(exp1, n, x))
    if exp1_val > 0:
        return math.log2(exp1_val)
    else:
        return 0

# returns e^ exp1
def exp(exp1, n, x):
    exp1_val = float(evaluate(exp1, n, x))

    return math.e ** exp1_val

#returns larger value of exp1 and exp2
def max(exp1, exp2, n, x) -> float:
    exp1_eval = float(evaluate(exp1, n, x))
    exp2_eval = float(evaluate(exp2, n, x))
    
    if exp1_eval > exp2_eval :
        return exp1_eval
    else:
        return exp2_eval

# Returns exp3 if exp1 <= exp2, otherwise returns exp4
def ifleq(exp1, exp2, exp3, exp4, n, x):
    if float(evaluate(exp1, n, x)) <= float(evaluate(exp2, n, x)):
        return evaluate(exp3, n, x)
    else:
        return evaluate(exp4, n, x)

# Returns the element at exp1th element (floored, absed and modded)
def data(exp1, n, x) -> float:
    eval = float(evaluate(exp1, n, x))
    floor = math.floor(eval)
    index = np.abs(floor)
    if not index == 0:
        index = index % n 
    return x[index]

# Returns the difference between 2 elements
def diff(exp1, exp2, n, x) -> float:
    exp1_data = data(exp1, n, x)
    exp2_data = data(exp2, n, x)
    return exp1_data - exp2_data

# Returns the average of a range between 2 indecies
def avg(exp1, exp2, n, x):

    k = np.abs(math.floor(evaluate(exp1, n, x))) % n
    l = np.abs(math.floor(evaluate(exp2, n, x))) % n
    print("k , l" , k , l)
    if k == l:
        return 0
    
    difference = np.abs(k - l) + 1
    factor = 1/difference
    sum = 0
    
    if k < l: #exp1 to exp2
        for i in range(k, l+1):
            sum += data(i)
    else: # exp2 to exp1
        for i in range(l, k):
            sum += data(i)
            
    return float(factor * sum)

# Evaluates an s-expression with input vector x of dimension n
def evaluate(sexp, n: int, x: float) -> float:
    # if its an atom
    if isinstance(sexp, int):
        return sexp
    elif isinstance(sexp, sex.Symbol):
        # Handle symbols?
        print("its a symbol", sexp)
        return None
    elif isinstance(sexp, list):
    
        operator = str(sex.car(sexp))
        operands = sex.cdr(sexp)
        #print("operator", operator,"operands",  operands)
        try:
            if operator == 'add':
                return add(*operands, n, x)
            elif operator == 'sub':
                return sub(*operands, n, x)
            elif operator == 'mul':
                return mul(*operands, n, x)
            elif operator == 'div':
                return div(*operands, n, x)
            elif operator == 'pow':
                return pow(*operands, n, x)
            elif operator == 'sqrt':
                return sqrt(*operands, n, x)
            elif operator == 'log':
                return log(*operands, n, x)
            elif operator == 'exp':
                return exp(*operands, n, x)
            elif operator == 'max':
                return max(*operands, n, x)           
            elif operator == 'ifleq':
                return ifleq(*operands, n, x)
            elif operator == 'data':
                return data(*operands, n, x)
            elif operator == 'diff':
                return diff(*operands, n, x)
            elif operator == 'avg':
                return avg(*operands, n, x)
        except OverflowError:
            print("Value too large for operation: ", operator, operands, " returning 0 for this step")
            return 0

# ------------
# QUESTION 2
# ------------

# Opens the training data file and reads the contents into the 2 respective lists
def open_training_data(training_data):
    x_values = []
    y_values = []
    # open file
    with open(training_data, 'r') as training_data_file:
        for line in training_data_file: 
            # split by tab
            values = line.split('\t')
            x_values.append(values[:-1])
            y_values.append(values[-1])
            
    return x_values, y_values

# Calculates the squared error between the evaulation of a s-expression and the output value y
def squared_error(sexp, y, n, x):
    difference = y - evaluate(sexp, n, x)
    return difference ** 2

# Calculated the mean squared error of an expression 
def calculate_fitness(e, n, m, training_data):
    x_values, y_values = open_training_data(training_data)
    total = 0
    factor = 1/m
    for i in range(0, m-1):
        x = x_values[i]
        y = y_values[i]
        total += squared_error(e, y, n, x)
            
    return(factor * total)

# ------------
# QUESTION 3
# ------------



# Branch swap - swap 2 branches from parents
def branch_swap_crossover(parent1: str, parent2: str, tree_depth: int, min_depth: int):
    branch_depth = random.randint(min_depth, tree_depth) # from 1 to avoid replacing whole tree
    branch = get_branch_at_depth(parent2, branch_depth) # select 2 random branches
    #swap branches

    try:
        parent1 = parent1.replace(get_branch_at_depth(parent1, branch_depth), branch, 1) 
    except AttributeError:
        print("wrongAttribute in crossover")
    return parent1

# Performs the branch swap crossover for a list of parents
def crossover(parents: list, tree_depth: int, min_depth: int, offspring_size: int):
    offspring = []
    # pick 2 parents
    for _ in range(offspring_size):
        parent1 = parents[random.randint(0,offspring_size-1)]
        parent2 = parents[random.randint(0,offspring_size-1)]
        offspring.append(branch_swap_crossover(parent1, parent2, tree_depth, min_depth))

    return offspring

# Branch replacement - Pick a random branch, replace with newly generated branch of same depth
def branch_replacement_mutation(parent: str, treedepth: int) -> str:
    branch_depth = random.randint(1, treedepth-1)
    branch = get_branch_at_depth(parent, branch_depth)
    replacement = full_generation(treedepth - branch_depth)
    new =  parent.replace(branch, replacement , 1)
    return new


# Performs mutation on the parents with a given probability
def mutation(parents: list, treedepth: int, mutation_rate: float) -> list:
    for parent in parents:
        if random.uniform(0,1) < mutation_rate:
            parent = branch_replacement_mutation(parent, treedepth)
            try:
                if ')(' in parent:
                    parent = parent.replace(')(', ') (')
            except TypeError:
                print("fortnite")
    return parents
    
# Returns a random branch at a given depth
def get_branch_at_depth(exp: str, depth: int) -> str:
    # remove leading and trailing ()
    exp = exp[1:-1]
    temp_exp = exp
    current_depth = 0
    for i, char in enumerate(exp):
        if current_depth == depth:
            return temp_exp
            
        else:
            if char == '(':
                current_depth +=1
                temp_exp = find_balanced_expression(exp[i:])
                
    return temp_exp

# Returns a list of all sub-expressions at the uppermost level    
def find_balanced_expression(exp: str) -> str:
    # monitor number of l and r brackets
    l_brac, r_brac = 0, 0
    start = 0
    expressions = []
    
    for i, char in enumerate(exp):
        if char == '(':
            l_brac +=1
            
        if char == ')':
            r_brac +=1
            
        if r_brac > l_brac: # if a close has been read before an open, ignore it
            r_brac = 0
            
        if l_brac == r_brac and l_brac > 0: # If brackets are present and balanced 
            expressions.append(exp[start:i+1].strip()) # add it to list, removing whitespace
            start = i + 2 # check next part of list
            l_brac, r_brac = 0,0 # reset bracket counters
    if expressions == []:
        print("No expressions found")
    return random.choice(expressions)

# Performs tournamnt selection on a population
def tournament_selection(population: list, n: int, offspring_size: int, population_size: int) -> list:

    offspring = []
    for i in range(offspring_size):
        tournament = []
        for j in range(n):
            tournament.append(population[random.randint(0,population_size-1)])
        tournament = sorted(tournament, key=lambda x: calculate_fitness(x))
        # parent with lowest mse
        offspring.append(tournament[0])
    
    return offspring

# Generates a single member of population using full generation
def full_generation(tree_depth: int) -> str:
    if tree_depth == 0:
        # At the leaf level, generate a random function or data node
        return f"{random.randint(1, 10)}"
    else:
        # Choose a random function node with its children
        node_name, num_children = random.choice(function_nodes)
        children = ' '.join(full_generation(tree_depth - 1) for _ in range(num_children))
        return f"({node_name} {children})"

# Generates a population of given size - Ramped half and half
def generate_population(population_size: int, tree_depth: int) -> list:
    population = []
    for _ in range(population_size):
        population.append(full_generation(tree_depth))
    return population

# Replaces less fit individuals in the current population with the offspring
def reproduction(population: list, offspring: list, offspring_size: int):
    population = sorted(population, key=lambda x : calculate_fitness(x))
    population[-offspring_size:] = offspring
    return population

def ga(params: list, inputs:list):
    tree_depth, crossover_n, offspring_size, mutation_rate = params
    
    start_time = time.time()
    elapsed_time = 0
    population_size, n, m, training_data, time_budget = inputs
    population = generate_population(population_size, tree_depth)
    
    while elapsed_time < time_budget:
        # Selection
        parents = tournament_selection(population, crossover_n, offspring_size, population_size)
        # Variation
        parents = mutation(parents, tree_depth, mutation_rate)
        
        offspring = crossover(parents, tree_depth, 2, offspring_size)
            # Fitness Calculation
            # Fitnesses are not maintained, but calculated when required
        # Reproduction
        population = reproduction(population, offspring, offspring_size)
        elapsed_time = time.time() - start_time
    
    print(elapsed_time)

    print(len(population))
    return population[0]

# ------------
# PROGRAM FLOW
# ------------
# Processes various arguments
def get_args(args) -> list:
    # Access the arguments
    # q1+
    question = getattr(args, 'question', None)
    expr = getattr(args, 'expr', None)
    n = getattr(args, 'n', None)
    x = getattr(args, 'x', None)
    # q2+
    m = getattr(args, 'm', None)
    data = getattr(args, 'data', None)
    # q3+
    pop_size = getattr(args, 'lambda', None)
    time_budget = getattr(args, 'time_budget', None)
    return question, expr, n, x, m, data, pop_size, time_budget

#
def select_question(args):
    
    # Extract arguments
    q, e, n, x, m, training_data, pop_size, time_budget  = get_args(args)

    if q == 1:
        # cast arguments
        e = sex.loads(e)
        n = int(n)
        x = [float(num) for num in x.split(' ')]
        result = evaluate(e)
        print(result)
    if q == 2:
        # cast arguments
        e = sex.loads(e)
        n = int(n)
        m = int(m)
        result = calculate_fitness(e, n, m, training_data)
        
    if q == 3:
        pop_size = int(pop_size)
        time_budget = float(time_budget)
        result = ga()

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse')

    # Add arguments
    parser.add_argument('-question', help='question number')
    parser.add_argument('-expr', help='an expression')
    parser.add_argument('-n', help='dimension of unput vector n')
    parser.add_argument('-x', help='input vector')
    
    parser.add_argument('-m', help='Size of training data')
    parser.add_argument('-data', help='filename containing training data')
    
    parser.add_argument('-lambda', help='Population Size')
    parser.add_argument('-time_budget', help='number of seconds to run algorithm')
    
    # Parse the command line arguments
    args = parser.parse_args()
    select_question(args)
    
    return 

if __name__ == "__main__":
    print(evaluate(sex.loads(SAMPLE_EXP_1), 1, 1))
    print(evaluate(sex.loads(SAMPLE_EXP_2), 2, [1,2]))
    main()