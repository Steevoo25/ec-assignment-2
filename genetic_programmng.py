import time
import random
import sexpdata as sex

SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"

#function_nodes = [('add', 2),('sub', 2),('mul', 2),('div', 2),('pow', 2),('sqrt', 1),('log', 1),('exp', 1),('max', 2),('ifleq', 4),('data', 1),('diff', 2),('avg', 2)]

# Branch swap - swap 2 branches from parents
def crossover(parent1, parent2, branch_index):
    return 1
# Branch replacement - Pick a random branch, replace with newly generated branch
def mutation(parent):
    mutated = parent.copy()
    # select random branch and delete it
    # generate new branch and add it where branch was deleted from
    return 

# Generates a single member of population using full generation
def full_generation(tree_depth):
    
    return 1

# Generates a single member of population using growth generation
def growth_generation(max_depth):
    
    return 1

def tournament_selection(population, fitnesses, n, offspring_size):
    return 1
    

# Generates a population of given size - Ramped half and half
def generate_population(population_size, max_depth):
    population = []
    for _ in range(population_size):
        #population.append(generate_solution(max_depth))
        population = population
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
    population = generate_population(population_size)
    fitnesses = calclulate_fitnesses(population)
    
    while time_elapsed > time_budget:
        
        # Selection
        # Variation
        # Fitness Calculation
        # Reproduction
        time_elapsed +=1
    return 1
    
sexp = sex.loads(SAMPLE_EXP_1)
print(sexp)