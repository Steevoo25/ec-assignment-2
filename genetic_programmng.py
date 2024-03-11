
# Branch swap
def crossover():
    return 1

def mutation():
    return 1

# Generates a population of given size
def generate_population(population_size):
    population = []
    for _ in range(population_size):
        population.append(generate_solution())
    return population

def calculate_fitness():
    return 1

def ga(population_size: int, time_budget: int):
    return 1