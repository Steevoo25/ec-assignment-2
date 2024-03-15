from hjs115 import ga, open_training_data
import optuna
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # Test different parameter setting and collect the data (plot)
    columns = ["solution", "fitness", "tree_depth", "tournament_n", "offspring_size", "mutation_rate", "penalty_weight", "population_size"]
    
    training_x, training_y = open_training_data()
    
    # Initialise df columns
    df = pd.DataFrame(columns=columns)

    # Returns fitness of SA with randomly generated parameters
    def objective(trial):
        tree_depth = 0
        tournament_n = 0
        offspring_size = 0
        mutation_rate = 0
        params = [tree_depth, tournament_n, offspring_size, mutation_rate]
        pop_size = 0
        budget = 0
        inputs = [pop_size, n, m, training_x, training_y, budget]
        # Define parameter ranges
        #time budget range = 20-60s
        # Perform SA
        sol, fitness = ga(params=params, inputs=inputs)
        # Store results in df
        df.loc[trial.number] = (sol, fitness, *params)
        return fitness

    # Tune parameters
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    # Print best results
    print(study.best_params)

    # Show whole dataframe
    pd.set_option('display.max_rows', None)

    # Show all results
    df = df.sort_values(by='fitness')
    df.index.name = 'Index'
    print(df)

    # print mean and standard deviations
    print(df.mean())
    print(df.std())
    
    # Generate box plots
    
    #observe results