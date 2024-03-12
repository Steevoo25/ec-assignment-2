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

# Branch swap - swap 2 branches from parents
def crossover(parent1, parent2, branch_index: int):
    return 1

# Branch replacement - Pick a random branch, replace with newly generated branch
def mutation(parent):
    # a branch is a well-bracketed string
    # get a random element
    mutated = parent[random.randint(0,len(parent)):]
    
    print(mutated)
    # select random branch and delete it
    # generate new branch and add it where branch was deleted from
    return mutated


def tournament_selection(population: list, fitnesses: list, n: int, offspring_size: int):
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

def calculate_fitness(solution):
    return 1
    
def calclulate_fitnesses(population: list):
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

def get_branch_at_depth(exp: str, depth: int):
    # remove leading and trailing ()
    exp = exp[1:-1]
    current_depth = 0
    for i, char in enumerate(exp):
        if current_depth == depth:
            current_depth = depth
            
        else:
            if char == '(':
                current_depth +=1
                exp = find_next_close_bracket(exp[i:])
            if char == ')':
                print("Going shallower")
    return exp
    
def find_next_close_bracket(exp):
    for i, char in enumerate(exp):
        if char == ')':
            return exp[:i+1]
            
print(get_branch_at_depth(SAMPLE_EXP_1, 1))
print(get_branch_at_depth(SAMPLE_EXP_1, 0))
    
#ga(1,1,3)
