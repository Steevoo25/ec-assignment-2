import time
import random
import sexpdata as sex
from treelib import Node, Tree
from s_expressions import evaluate

function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4),('data', 1),('diff', 2),('avg', 2)]
function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4)] #,('data', 1),('diff', 2),('avg', 2)]


SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"

# Branch swap - swap 2 branches from parents
def crossover(parent1, parent2, branch_index):
    return 1
# Branch replacement - Pick a random branch, replace with newly generated branch
def mutation(parent):
    mutated = parent.copy()
    # select random branch and delete it
    # generate new branch and add it where branch was deleted from
    return 1

def tournament_selection(population, fitnesses, n, offspring_size):
    # for offspring_size times
        # pick n random solutions from population
        # copy one with highest fitness into offspring
    
    return 1

# Generates a single member of population using full generation
def full_generation(depth) -> str:
    if depth == 0:
        # At the leaf level, generate a random function or data node
        return f"{random.randint(1, 10)}"
    else:
        # Choose a random function node with its children
        node_name, num_children = random.choice(function_nodes)
        children = ' '.join(full_generation(depth - 1) for _ in range(num_children))
        return f"({node_name} {children})"

# Generates a population of given size - Ramped half and half
def generate_population(population_size, max_depth):
    population = []
    for _ in range(population_size):
        population.append(full_generation(max_depth))
    return population

def calculate_fitness(solution):
    return 1
    
def calclulate_fitnesses(population):
    fitnesses = []
    for solution in population:
        fitnesses.append(calculate_fitness(solution))
    return fitnesses

def ga(population_size: int, time_budget: int):
    time_elapsed = 0
    population = generate_population(population_size, 2)
    
    print(SAMPLE_EXP_1)
    print(sex.loads(SAMPLE_EXP_1))
    
    sexp = full_generation(2)
    sexp = sex.loads(sexp)
    print(evaluate(sexp))
    fitnesses = calclulate_fitnesses(population)
    
    while time_elapsed > time_budget:
        # Selection
        # Variation
        # Fitness Calculation
        # Reproduction
        time_elapsed +=1
    return 1

ga(1,2)