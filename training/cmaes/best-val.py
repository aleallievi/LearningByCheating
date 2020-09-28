import numpy as np
import os
import pandas as pd


def get_latest_subdir(parent_dir_path='.'):
    """
    Returns most recently created directory in given path
    """
    subdirs = []
    for subdir in os.listdir(parent_dir_path):
        subdirs.append(os.path.join(parent_dir_path, subdir))

    dir = max(subdirs, key=os.path.getmtime)
    return dir


path = '/home/boschaustin/projects/CL_AD/ES/carla_lbc_new/cma_results/'
latest_results_dir = get_latest_subdir(path)
result_file = os.path.join(latest_results_dir, 'results/value_score.txt')
result_df = pd.read_csv(result_file, sep=' ', header=None, names=['Gen', 'Indiv', 'Fit'])

# Print the mean of the top 5 scores for each generation
for gen in range(max(result_df['Gen'])):
    print(gen, ' ', ((result_df[result_df['Gen'] == gen]['Fit']).nlargest(5)).mean())

