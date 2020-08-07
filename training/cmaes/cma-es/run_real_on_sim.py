"""Get best performing parameters from CMA-ES optimization."""
import os
import subprocess as sp
from shutil import copyfile
import argparse
import matplotlib.pyplot as plt
from get_best import get_best, create_params_dir, params_dir_exists, get_params_dir_name

parser = argparse.ArgumentParser()
parser.add_argument('--upper_limit', default=None, type=int,
                    help='Maximum generation to include.')
parser.add_argument('--result_dir', help='Path to parameter and value files.', type=str, default=None)
parser.add_argument('--real_params', help='Params to run for the executable, models reality', type=str, default=None)
parser.add_argument('--executable', help='path to the executable', type=str, default=None)
parser.add_argument('--save_at', help='path to save the generated figure', type=str, default=None)
flags = parser.parse_args()

def main():  # noqa
    if not params_dir_exists(flags.result_dir):
        print('results dir does not exist, creating')
        best, values = get_best(flags.result_dir, flags.upper_limit)
        create_params_dir(best, values, flags.result_dir)
    ctr = 1
    params_dir = get_params_dir_name(flags.result_dir)
    params_files = os.listdir(params_dir)
    values = []
    while True:
        files = [f for f in params_files if f.startswith('params_%s_i' % str(ctr))]
        if len(files) == 0:
            break
        assert len(files) == 1, 'There should be exactly one file like this'
        this_gen_params = files[0]
        value = float(sp.check_output([flags.executable, '/dev/null', '--sim_params_file', flags.real_params, '--params_file', params_dir + '/' + this_gen_params, '--stdout']))
        values += [value]
        ctr += 1

    plt.figure(figsize=(12, 9))
    plt.title('Fitness vs Num Generations, sim optimization on real domain', fontsize=25)
    plt.xlabel('# Generations', fontsize=25)
    plt.ylabel('Fitness', fontsize=25)
    plt.plot(range(len(values)), values, linewidth=2)
    plt.legend(fontsize=22, fancybox=True).get_frame().set_alpha(0.5)
    fig = plt.gcf()
    if flags.save_at is not None:
        fig.savefig(flags.save_at)
    plt.show()


if __name__ == '__main__':
    main()
