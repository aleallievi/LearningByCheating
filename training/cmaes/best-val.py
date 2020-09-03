import numpy as np
import os

path = '/home/boschaustin/projects/CL_AD/ES/carla_lbc/cma_results/results/'
returns = []
for filename in os.listdir(path):
    if filename.split('_')[0] == 'value':
        with open(path+filename) as filepath:
            returns.append(float(filepath.readline()))

print('All returns: ', returns)
print('# of individuals: ', len(returns), 'Best return: ',  np.max(returns))

