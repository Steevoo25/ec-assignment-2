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
SAMPLE_GA_PARAMS = [4, 2, 2, 0.1, 4]
HIGH_FITNESS = 10_000
#tree_depth, tournament_n, offspring_size, mutation_rate, penalty_weight

FUNCTION_NODES = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4),('data', 1),('diff', 2),('avg', 2)]
MIN_CROSSOVER_DEPTH = 1

# ------------
# QUESTION 1
# ------------
# adds 2 expressions
def add(exp1, exp2, n, x) -> float:
    return float(evaluate(exp1, n, x)) + float(evaluate(exp2, n, x))

# Subtracts exp2 from exp1
def sub (exp1, exp2, n, x) -> float:
    return float(evaluate(exp1, n, x)) - float(evaluate(exp2, n, x))
    
# Multiplies 2 expressions
def mul(exp1, exp2, n, x) -> float:
    return float(evaluate(exp1, n, x)) * float(evaluate(exp2, n, x))

# Divides exp1 by exp 2 as long as exp2 is not 0
def div(exp1, exp2, n, x) -> float:
    exp2_eval = float(evaluate(exp2, n, x))
    exp1_eval = float(evaluate(exp1, n, x))
    if exp2_eval == 0:
        return 0
    else:
        return exp1_eval / exp2_eval
        
# Returns exp1 to the power of exp2
def pow(exp1, exp2, n, x) -> float:
    
    exp1_eval = float(evaluate(exp1, n, x))
    exp2_eval = float(evaluate(exp2, n, x))
    if exp1_eval == 0 and exp2_eval < 0:
        return 0
    else:
        return float(evaluate(exp1, n, x)) ** float(evaluate(exp2, n, x))

# Returns the square root of exp1
def sqrt(exp1, n, x) -> float:
    exp1_eval = float(evaluate(exp1, n, x))
    if exp1_eval < 0:
        return 0
    else:
        return math.sqrt(exp1_eval)

# Takes log2 of 1 expression
def log(exp1, n, x) -> float:
    exp1_val = float(evaluate(exp1, n, x))
    if exp1_val > 0:
        return math.log2(exp1_val)
    else:
        return 0

# returns e^ exp1
def exp(exp1, n, x) -> float:
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
def ifleq(exp1, exp2, exp3, exp4, n, x) -> float:
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
    exp1_eval = evaluate(exp1)
    exp2_eval = evaluate(exp2)
    
    exp1_data = data(exp1_eval, n, x)
    exp2_data = data(exp2_eval, n, x)
    
    return exp1_data - exp2_data

# Returns the average of a range between 2 indecies
def avg(exp1, exp2, n, x) -> float:
    k = np.abs(math.floor(evaluate(exp1, n, x))) % n
    l = np.abs(math.floor(evaluate(exp2, n, x))) % n
    if k == l:
        return 0
    
    difference = np.abs(k - l) + 1
    factor = 1/difference
    sum = 0
    
    if k < l: #exp1 to exp2
        for i in range(k, l+1):
            sum += data(i, n, x)
    else: # exp2 to exp1
        for i in range(l, k + 1):
            sum += data(i, n, x)
            
    return float(factor * sum)

# Evaluates an s-expression with input vector x of dimension n
def evaluate(sexp, n: int, x: list) -> float:
    # if its an atom
    if isinstance(sexp, int):
        return sexp
    elif isinstance(sexp, float):
        return sexp
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
            #print("Value too large for operation: ", operator, operands, " returning 0 for this step")
            return 0
        except TypeError:
            #print("Type error found:", operator, operands, type(operator), type(operands))
            return 0
        except Exception:
            return 0

# ------------
# QUESTION 2
# ------------

# Opens the training data file and reads the contents into the 2 respective lists
def open_training_data(training_data: str):
    x_values = []
    y_values = []
    # open file
    with open(training_data, 'r') as training_data_file:
        for line in training_data_file: 
            # split by tab
            values = line.split('\t')
            x_values.append([float(x) for x in values[:-1]]) # x values become a list of list[float]
            y_values.append(float(values[-1])) # y values becomes a list of float
    return x_values, y_values

# Calculates the squared error between the evaulation of a s-expression and the output value y
def squared_error(sexp, y: float, n: int, x: list):
    try:
        difference = (y - evaluate(sexp, n, x)) ** 2
    except OverflowError:
        difference = math.inf
    except TypeError:
        print(sexp)
    return difference

# Calculated the mean squared error of an s-expression e 
def calculate_fitness(e, n: int, m: int, training_x:list, training_y:list) -> float:
    total = 0
    factor = 1/m
    for i in range(0, m-1):
        x = training_x[i]
        y = training_y[i]
        total += squared_error(e, y, n, x)
            
    return(factor * total)

# ------------
# QUESTION 3
# ------------

# returns the arity of a given branch
def check_arity(branch: str) -> int:
    # get function name form branch
    start = branch.find('(') +1
    end = branch.find(' ')
    node_name = branch[start:end]
    for node in FUNCTION_NODES:
        if node[0] == node_name:
            return node[1]
    return False

# Branch swap - swap 2 branches from parents
def branch_swap_crossover(parent1: str, parent2: str, tree_depth: int, min_depth: int):
    branch_depth = random.randint(min_depth, tree_depth) # from 1 to avoid replacing whole tree
    branch1 = get_branch_at_depth(parent1, branch_depth) # select 2 random branches
    branch2 = get_branch_at_depth(parent2, branch_depth)
    # check arity of branches is the same
    while not check_arity(branch1) == check_arity(branch2):
        branch1 = get_branch_at_depth(parent1, branch_depth)
        branch2 = get_branch_at_depth(parent2, branch_depth)
    #swap branches
    
    parent1 = parent1.replace(get_branch_at_depth(parent1, branch_depth), branch2, 1) 
    parent2 = parent2.replace(get_branch_at_depth(parent2, branch_depth), branch1, 1)
    
    if ')(' in parent1:
        parent1 = parent1.replace(')(', ') (')
    if ')(' in parent2:
        parent2 = parent2.replace(')(', ') (')
    
    return parent1, parent2

# Performs the branch swap crossover for a list of parents
def crossover(parents: list, tree_depth: int, min_depth: int, offspring_size: int):
    offspring = []
    # pick 2 parents
    for _ in range(offspring_size//2):
        parent1 = parents[random.randint(0,offspring_size-1)]
        parent2 = parents[random.randint(0,offspring_size-1)]
        new_offspring = branch_swap_crossover(parent1, parent2, tree_depth, min_depth)
        offspring.append(new_offspring[0])
        offspring.append(new_offspring[1])

    return offspring

# Branch replacement - Pick a random branch, replace with newly generated branch of same depth
def branch_replacement_mutation(parent: str, treedepth: int) -> str:
    branch_depth = random.randint(1, treedepth-1)
    branch = get_branch_at_depth(parent, branch_depth)
    replacement = full_generation(treedepth - branch_depth)
    new =  parent.replace(branch, " " + replacement , 1)
    return new

# Performs mutation on the parents with a given probability
def mutation(parents: list, treedepth: int, mutation_rate: float) -> list:
    for parent in parents:
        if random.uniform(0,1) < mutation_rate:
            parent = branch_replacement_mutation(parent, treedepth)
            if ')(' in parent:
                parent = parent.replace(')(', ') (')
    return parents
    
# Returns a random branch at a given depth
def get_branch_at_depth(exp: str, depth: int) -> str:
    # remove leading and trailing ()
    exp = exp[1:-1]
    temp_exp = exp
    current_depth = 0
    for i, char in enumerate(exp):
        if current_depth == depth:
            if temp_exp == []:
                return get_branch_at_depth(exp, depth-1) # if nothing has been found, look one layer up
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
            expressions.append(exp[start:i+1]) # add it to list, removing whitespace
            start = i + 2 # check next part of list
            l_brac, r_brac = 0,0 # reset bracket counters
    if expressions == []:
        #print("No expression found", exp)
        return []
    return random.choice(expressions)

# Performs tournamnt selection on a population
def tournament_selection(population: list, fitnesses: list, tournament_n: int, offspring_size: int, population_size: int, n, m, training_x, training_y, penalty_weight) -> list:
    offspring = []
    for i in range(offspring_size):
        tournament = []
        for j in range(tournament_n):
            tournament.append(random.randint(0,population_size-1)) # append indexes
        best = sorted(tournament, key=lambda x: fitnesses[x]) # sort by fitness value
        # parent with lowest mse
        offspring.append(population[0])
    
    return offspring

# Generates a single member of population using full generation
def full_generation(tree_depth: int) -> str:
    if tree_depth == 0:
        # At the leaf level, generate a random function or data node
        return f"{random.randint(1, 10)}"
    else:
        # Choose a random function node with its children
        node_name, num_children = random.choice(FUNCTION_NODES)
        children = ' '.join(full_generation(tree_depth - 1) for _ in range(num_children))
        return f"({node_name} {children})"

# Generates a population of given size - Ramped half and half
def generate_population(population_size: int, tree_depth: int) -> list:
    population = []
    for _ in range(population_size):
        population.append(full_generation(tree_depth))
    return population

# Replaces less fit individuals in the current population with the offspring
def reproduction(population: list, offspring: list,fitnesses: list, offspring_fitnesses:list, offspring_size: int, pop_size: int,n: int, m: int, training_x, training_y, penalty_weight):
    
    ranked_fitnesses = []
    sorted_fitnesses = sorted(fitnesses) # sort fitnesses
    # rank indexes by fitness
    for i in range(pop_size):
        fitness_index = fitnesses.index(sorted_fitnesses[i])
        ranking = [i,fitness_index]
        ranked_fitnesses.append(ranking)
    # replace lowest ranking indexes with offspring
    for i in range(offspring_size):
        replacement_index = ranked_fitnesses[pop_size - i - 1][1]
        try:
            population[replacement_index] = offspring[i-1] #replace solution at rank i, with offspring i
            fitnesses[replacement_index] = offspring_fitnesses[i-1] # replace fitnesses at i with offspring fitnesses at i
        except IndexError:
            print(i, len(offspring), offspring)
            #population[replacement_index] = offspring[i] #replace solution at rank i, with offspring i
            #fitnesses[replacement_index] = offspring_fitnesses[i] # replace fitnesses at i with offspring fitnesses at i
    return population, fitnesses

# Introduces a penalty for bloat
def bloat_penalty(e: str, penalty_weight: float=1) -> float:
    # fitness is being minimised so penalty will be positive
    
    return e.count('(') * penalty_weight
    
# Calculate fitness for a string e
def calculate_genetic_fitness(e:str,n: int, m: int, training_x: list, training_y: float, penalty_weight: float):
    penalty = bloat_penalty(e, penalty_weight)
    fitness = 0
    try:
        e = sex.loads(e)
    except sex.ExpectClosingBracket:
        return HIGH_FITNESS
    except sex.ExpectNothing:
        return HIGH_FITNESS
    fitness = calculate_fitness(e, n, m,  training_x, training_y)
    if isinstance(fitness, complex):
        fitness = fitness.real
    return fitness + penalty

# Performs genetic algorithm with parameters params and agruments inputs
def ga(params, inputs):
    start_time = time.time()
    # unpack parameters
    tree_depth, tournament_n, offspring_size, mutation_rate, penalty_weight = params
    population_size, n, m, training_x, training_y, time_budget = inputs
    #print("unpack")
    # generate initial population
    population = generate_population(population_size, tree_depth)
    #print("generate")
    fitnesses = [calculate_genetic_fitness(e, n, m, training_x, training_y, penalty_weight) for e in population]
    #print("fitness")
    # initialise timer
    elapsed_time = time.time()
    
    while elapsed_time < time_budget:
        # Selection
        parents = tournament_selection(population, fitnesses, tournament_n, offspring_size, population_size, n, m, training_x, training_y, penalty_weight)
        #print("selection")
        # Variation
        parents = mutation(parents, tree_depth, mutation_rate)
        offspring = crossover(parents, tree_depth, MIN_CROSSOVER_DEPTH, offspring_size)
        #print("variation")
        
        # Fitness Calculation
        offspring_fitnesses = [calculate_genetic_fitness(e, n, m, training_x, training_y, penalty_weight) for e in offspring]
        #print("fitness calc")
        # Reproduction
        population, fitnesses = reproduction(population, offspring, fitnesses, offspring_fitnesses, offspring_size, population_size, n, m, training_x, training_y, penalty_weight)
        #print("reprod")
        #print(len(population))
        elapsed_time = time.time() - start_time
    
    sorted_list = sorted(list(zip(population, fitnesses)), key= lambda x : x[1])
    return sorted_list[0]
    
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

# Executes correspoding functionality for each question
def select_question(args):
    
    # Extract arguments
    q, e, n, x, m, training_data, pop_size, time_budget = get_args(args)

    if q == '1':
        # cast arguments
        e = sex.loads(e)
        n = int(n)
        x = [float(num) for num in x.split(' ') if not num == '']
        result = evaluate(e, n, x)

    if q == '2':
        # cast arguments
        e = sex.loads(e)
        n = int(n)
        m = int(m)
        training_x, training_y = open_training_data(training_data) # open file
        #training_x = [float(num) for num in training_x.split(' ')]
        result = calculate_fitness(e, n, m, training_x, training_y)
        
    if q == '3':
        pop_size = int(pop_size)
        n = int(n)
        m = int(m)
        time_budget = float(time_budget)
        training_x, training_y = open_training_data(training_data) # open file
        #training_x = [float(num) for num in training_x.split(' ')]
        params = SAMPLE_GA_PARAMS # [0] = tree depth, [1] = tournament_n, [2] = offspring_size, [3] = mutation_rate
        inputs = [pop_size, n, m, training_x, training_y, time_budget] # [0] = population_size, [1] = n, [2] = m, [3] = training_data, [4] = time_budget
        result = ga(params=params, inputs=inputs)
        
    print(result)

#Parses arguments and runs main logic function
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

# Entry point
if __name__ == "__main__":
    error_exp = "(pow (sub (log (mul 4 10)) (ifleq (data 10) (add 9 4) (exp 9) (pow 10 9))) (add (exp (ifleq 1 3 1 10)) (ifleq (max 3 9) (div 6 5) (div 4 8) (add 10 7))))"
    error_exp = "(max (exp (ifleq (max 5 3) (log 5) (pow 5 4) (log 3)) (log 3)))"
    e = sex.loads(error_exp)
    #print("error?", evaluate(e, 1,[1]))
    main()