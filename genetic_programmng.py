import time
import random
import sexpdata as sex
from treelib import Tree
from s_expressions import evaluate

function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4),('data', 1),('diff', 2),('avg', 2)]
#function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4)] #,('data', 1),('diff', 2),('avg', 2)]
#function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('sqrt', 1),('log', 1),('max', 2),('ifleq', 4)] #,('data', 1),('diff', 2),('avg', 2)]


SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"
SAMPLE_EXP_3 = "(max (sub (mul 2 3) (add 1 1)) (exp (add 4 6)))"

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
    try:
        return parent.replace(branch, replacement , 1)
    except AttributeError:
        print("wrongAttribute in mutation")

# Performs mutation on the parents with a given probability
def mutation(parents: list, treedepth: int, mutation_rate: float) -> list:
    for parent in parents:
        if random.uniform(0,1) < mutation_rate:
            parent = branch_replacement_mutation(parent, treedepth)
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

# Performs tournamnt selection on a population
def tournament_selection(population: list, n: int, offspring_size: int, population_size: int) -> list:

    offspring = []
    for _ in range(offspring_size):
        tournament = []
        for _ in range(n):
            tournament.append(population[random.randint(0,population_size-1)])
        tournament = sorted(tournament, key=lambda x: calculate_fitness(x))
        # parent with lowest mse
        offspring.append(tournament[0])
    
    return offspring

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
    
def calculate_fitnesses(population: list) -> list:
    fitnesses = []
    for solution in population:
        fitnesses.append(calculate_fitness(solution))
    return fitnesses

# Replaces less fit individuals in the current population with the offspring
def reproduction(population: list, offspring: list, offspring_size: int):
    population = sorted(population, key=lambda x : calculate_fitness(x))
    print(len(population))
    print(offspring_size, len(offspring))
    population[-offspring_size:] = offspring
    print(len(population))
    return population

def ga(time_budget: int, population_size: int, tree_depth: int, crossover_n: int, offspring_size: int, mutation_rate: int):

    start_time = time.time()
    elapsed_time = 0
    
    population = generate_population(population_size, tree_depth)
    counter = 0
    while counter < 1:#elapsed_time<time_budget:
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
        counter +=1
    #for p in population:
        #print("--------------")
        #print(p)
    return population[0]
    #return population[0] # at this time the list will be sorted as reproduction has just occurred, which sorts the list
ga(0.01,10,4,2,3,0.2)

# print(SAMPLE_EXP_3)
# print(mutation(SAMPLE_EXP_3, 3))
# print(mutation(SAMPLE_EXP_3, 3))
            
#ga(1,1,3)
