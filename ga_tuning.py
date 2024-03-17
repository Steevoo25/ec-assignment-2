from hjs115 import ga, open_training_data
import optuna
import pandas as pd

SAMPLE_GA_PARAMS = [5, 2, 5, 0.1, 1.1] #tree_depth, tournament_n, offspring_size, mutation_rate, penalty_weight

DATA_PATH = './data/cetdl1772small.dat'
N = 13
M = 999


if __name__ == "__main__":

    # Test different parameter setting and collect the data (plot)
    columns = ["solution", "fitness", "tree_depth", "tournament_n", "offspring_size", "mutation_rate", "penalty_weight", "population_size", "time_budget"]
    
    # Initialise df columns
    df = pd.DataFrame(columns=columns)

    # Returns fitness of SA with randomly generated parameters
    def objective(trial):
        pop_size = 100
        time_budget = 30
        # inputs
        training_x, training_y = open_training_data(DATA_PATH)
        #pop_size = trial.suggest_int('pop_size', 20, 200)
        #time_budget = trial.suggest_int('time_budget', 20, 60)
        inputs = [pop_size, N, M, training_x, training_y, time_budget]
        # params
        tree_depth, tournament_n, offspring_size, mutation_rate, penalty_weight = SAMPLE_GA_PARAMS
        #tree_depth = trial.suggest_int('tree_depth', 3,10)
        #tournament_n = trial.suggest_int('tournament_n', 2,10)
        #offspring_size = trial.suggest_int('offspring_size', 2, 20)
        #mutation_rate = trial.suggest_float('mutation_rate', 1/pop_size, 20/pop_size)
        #penalty_weight = trial.suggest_float('penalty_weight', 1, 4)
        params = tree_depth, tournament_n, offspring_size, mutation_rate, penalty_weight

        sol, fitness = ga(params=params, inputs=inputs)
        # Store results in df
        print(sol)
        df.loc[trial.number] = (sol,fitness, *params, pop_size, time_budget)
        df.to_csv("./data/time_budget.csv", index=True)
        return fitness

    # Tune parameters
    study = optuna.create_study(study_name='time_budget')
    study.optimize(objective, n_trials=100)

    # Print best results
    print(study.best_params)

    # Show whole dataframe
    pd.set_option('display.max_rows', None)

    # Show all results
    df = df.sort_values(by='fitness')
    df.index.name = 'Index'
    df.to_csv("./data/time_budget.csv", index=True)
    
    print(df)