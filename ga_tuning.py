import random
import time
from hjs115 import ga
import optuna
import sexpdata as sex
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # Test different parameter setting and collect the data (plot)
    columns = []

    # Initialise df columns
    df = pd.DataFrame(columns=columns)

    # Returns fitness of SA with randomly generated parameters
    def objective(trial):
        # Define parameter ranges
        #time budget range = 20-60s
        # Perform SA
        _, fitness = ga()
        # Store results in df
        df.loc[trial.number] = ""
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

    #observe results