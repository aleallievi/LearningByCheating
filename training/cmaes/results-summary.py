import numpy as np
import os
import sys

###
# calculates and prints statistics regarding each CMA-ES generation's performance
#
# works best with python3
###

path = '/home/boschaustin/projects/CL_AD/ES/carla_lbc/cma_results/results/'

returns = []
gen_returns = {}
gen_counts = {}
num_indivs = 0

for filename in os.listdir(path):
    if filename.split('_')[0] == 'value':
        #print(filename)
        num_indivs = num_indivs + 1
        
        # count return as a finished individual in the generation
        gen = int(filename.split('_')[1])
        if gen in gen_counts:
            gen_counts[gen] = gen_counts[gen] + 1
        else:
            gen_counts[gen] = 1

        # read return from file
        indiv_return = 0
        with open(path+filename) as filepath:
            indiv_return = float(filepath.readline())
        #returns.append(indiv_return)

        # add return to a generation-specific array in dict gen_returns
        if gen in gen_returns:
            gen_returns[gen].append(indiv_return)
        else:
            gen_returns[gen] = []

# print results
print('Generation counts: ', dict(sorted(gen_counts.items())))
gen_returns = dict(sorted(gen_returns.items())) # need to uncomment and fix this
#print(gen_returns)
print('\nGeneration returns: ')
all_gens_max = sys.float_info.min
for gen in gen_returns:
    #print(gen)
    this_gen_max = np.max(gen_returns[gen])
    all_gens_max = max(this_gen_max, all_gens_max)
    print('\ngen: ', gen, '; max return: ', this_gen_max)
    print('mean return: ', np.mean(gen_returns[gen]))
    print('stdev of return: ', np.std(gen_returns[gen]))
    print('returns: ', gen_returns[gen])
#print('All returns: ', returns)
print('\n# of individuals: ', num_indivs, '\n\nBest return: ',  all_gens_max)

