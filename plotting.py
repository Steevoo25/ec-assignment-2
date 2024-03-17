import pandas as pd
from matplotlib import pyplot as plt

tuning_params = ["tree_depth", "offspring_size", "mutation_rate", "penalty_weight", "population_size", "time_budget"]

for param in tuning_params:
    filepath = f"./data/{param}.csv" # get filename
    df = pd.read_csv(filepath) # open csv
    df = df[df['fitness'] < 4000]
    df = df['fitness'] # filter to fitness
    plt.boxplot(df)
    plt.title(f"Boxplot of fitnesses for {param}")
    plt.ylim(1000,4000)
    plt.savefig(fname=f'./data/plots/{param}')
    plt.show()