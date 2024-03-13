Generate initial population-
	Full Generation:
		while treedepth > 0:
			add random_function_node
		for empty_branch in tree:
			add random_terminal_node
Start timer
while timer < budget:
	perform selection - 
		Tournament selection:
			for offspring_size times:
				randomly select tournament_size elements in population
				calculate fitness of elements
				add fittest element to offspring
			return offspring
	perform variation - 
		Mutation - 
			Branch Relacement:
				generate new_branch
				replace random_branch in tree with new_branch
		Crossover - 
			Branch Swap:
				pick 2 random_parents
				pick random_branch in each random_parent
				swap random_branch_1 with random_branch_2

	perform Reproduction -
		Steady State Reproduction:
			calculate fitness of population
			replace offspring_size many worst elements in population with offspring

calculate fitness of each element in population
return element with lowest fitness



