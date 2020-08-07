#!/usr/bin/python
"""Plot CMA-ES run."""
import os
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('results_folder', help='where params from the run are stored', type=str)
#parser.add_argument('--title', default='Fitness over Number of Iterations', help='title of the graph', default='Fitness vs Num Generations', type=str)
flags = parser.parse_args()

def main():  # noqa

    data = {}

    for results_file_path in glob.glob(flags.results_folder + "/value*"):
        results_file = os.path.basename(results_file_path)
        gen, _, ind = results_file.split('_')[1:4]
        policyID = '%s_i_%s' % (gen, ind)

        gen, _, ind = policyID.split('_')
        ind = ind[:-4]
        gen = int(gen)

        with open(results_file_path) as f:
            value = f.readlines()[0].rstrip()

        if gen not in data:
            data[gen] = []
        data[gen].append(float(value))

    gens, means, maxs, stds = [], [], [], []
    sizes = []
    mx, mn = -1e8, 1e8
    for i, gen in enumerate(data):
        gens.append(gen)
        means.append(np.mean(data[gen]))
        stds.append(np.std(data[gen]))
        sizes.append(np.size(data[gen]))
        maxs.append(max(data[gen]))
        #if i >= 40:
        print('%d\t%f\t%f' % (gen, np.mean(data[gen]), max(data[gen])))
        if maxs[-1] > mx:
            mx = maxs[-1]
        if means[-1] < mn:
            mn = means[-1]

    ci = 1.96 * np.array(stds) / np.sqrt(np.array(sizes))
    plt.figure(figsize=(12, 9))
    plt.title('temp', fontsize=25)
    plt.xlabel('# Generations', fontsize=25)
    plt.ylabel('Fitness', fontsize=25)
    plt.errorbar(gens, means, yerr=ci, linewidth=2, label='Average')
    plt.plot(gens, maxs, linewidth=2, label='Best')
    plt.ylim([mn - 10, mx + 10])
    plt.legend(fontsize=22, fancybox=True, framealpha=0.3)
    plt.show()


if __name__ == '__main__':
    main()
