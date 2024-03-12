import signal # timer
import random
import sexpdata as sex
from treelib import Tree
from s_expressions import evaluate

function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4),('data', 1),('diff', 2),('avg', 2)]
function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4)] #,('data', 1),('diff', 2),('avg', 2)]
#function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('sqrt', 1),('log', 1),('max', 2),('ifleq', 4)] #,('data', 1),('diff', 2),('avg', 2)]


SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"
SAMPLE_EXP_3 = "(max (sub (mul 2 3) (add 1 1)) (exp (add 4 6)))"

# Branch swap - swap 2 branches from parents
def crossover(parent1: str, parent2: str, branch_index: int) -> str:
    return 1

# Branch replacement - Pick a random branch, replace with newly generated branch
def mutation(parent: str, treedepth: int) -> str:
    mutated = parent
    branch_depth = random.randint(1, treedepth-1)
    branch = get_branch_at_depth(mutated, branch_depth)
    replacement = full_generation(treedepth - branch_depth)
    return mutated.replace(branch, replacement , 1)

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
    
def find_balanced_expression(exp) -> str:
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
    
    return random.choice(expressions)


def tournament_selection(population: list, fitnesses: list, n: int, offspring_size: int) -> list:
    # for offspring_size times
        # pick n random solutions from population
        # copy one with highest fitness into offspring
    
    return 1

# Generates a single member of population using full generation
def full_generation(tree_depth) -> str:
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

def calculate_fitness(solution: str) -> float:
    return 1
    
def calclulate_fitnesses(population: list) -> list:
    fitnesses = []
    for solution in population:
        fitnesses.append(calculate_fitness(solution))
    return fitnesses

def ga(population_size: int, time_budget: int, tree_depth: int ):
    time_elapsed = 0
    population = generate_population(population_size, tree_depth)
    
    for solution in population:
        print(solution)
        print(evaluate(sex.loads(solution)))
        
    fitnesses = calclulate_fitnesses(population)
    
    while time_elapsed < time_budget:
        # Selection
        # Variation
        offspring = mutation(population)
        # Fitness Calculation
        # Reproduction
        time_elapsed +=1
    return 1

mutation(SAMPLE_EXP_3, 3)
            
#ga(1,1,3)
