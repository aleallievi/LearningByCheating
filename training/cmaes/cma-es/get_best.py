"""Get best performing parameters from CMA-ES optimization."""
import os
import subprocess as sp
from shutil import copyfile
import argparse


def get_best(result_dir, upper_limit=None):
    best = {}
    values = {}

    for filename in os.listdir(result_dir):
        if filename.startswith('value'):
            fullname = os.path.join(result_dir, filename)
            ind = int(filename.split('_')[1])
            if upper_limit is not None and ind > upper_limit:
                continue
            # if ind not in inds: continue
            cmd = 'cat %s' % fullname
            value = float(sp.check_output(cmd.split()))
            if ind not in best:
                best[ind] = value
                values[ind] = filename
            if value > best[ind]:
                best[ind] = value
                values[ind] = filename
    return best, values

def get_params_dir_name(result_dir):
    return '%s/tmp' % result_dir

def params_dir_exists(result_dir):
    return os.path.isdir(get_params_dir_name(result_dir))

def create_params_dir(best, values, result_dir):
    ind = 1
    while True:
        if ind not in values:
            break
        params = 'params_' + '_'.join(values[ind].split('_')[1:])
        params_full = os.path.join(result_dir, params)
        print('%d %s %s %f' % (ind, values[ind], params, best[ind]))
        if not os.path.exists(os.path.join(result_dir, 'tmp')):
            os.mkdir(os.path.join(result_dir, 'tmp'))

        copyfile(params_full, get_params_dir_name(result_dir) + "/" + (params))
        ind += 1

def main():  # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper_limit', default=None, type=int,
                        help='Maximum generation to include.')
    parser.add_argument('result_dir', help='Path to parameter and value files.')
    flags = parser.parse_args()
    if params_dir_exists(flags.result_dir):
        print("Directory with params exists")
        #return
    best, values = get_best(flags.result_dir, flags.upper_limit)
    create_params_dir(best, values, flags.result_dir)


if __name__ == '__main__':
    main()
