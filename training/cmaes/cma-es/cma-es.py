#!/usr/bin/env python
"""Run CMA-ES on generic task."""

# imports
from __future__ import print_function
from __future__ import division

import psutil
import sys
import argparse
import os
import subprocess
import time
import getpass
import glob
import shutil
import numpy as np
import cma
import random
import pdb
import pickle
import torch
import signal
import pandas as pd
# import nvidia_smi  # only used to get GPU usage stats
from datetime import datetime


# parse arguments from launch script, by default only paths: experiment_path, config_file
# and flags: --run_local, --pop_size are used
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('experiment_path', type=str,
                    help='Directory for results.')
parser.add_argument('--executable', default=None, type=str, help='Path to executable.')
#parser.add_argument('num_iters', help='Number of iterations.',type=int)
parser.add_argument('--pop_size', help='Population size.', type=int, default=-1)    # CMA-ES default pop-size is 25
parser.add_argument('config_file', type=str, help='Configuration variables.')
#parser.add_argument('--seed_file', help='Seed file for CMA-ES.', default=None)
#parser.add_argument('--start_iter', help='Iteration to start from.', type=int, default=1)
parser.add_argument('--run_local', help='Launch on condor or not',
                    action='store_true', default=False)
parser.add_argument('--load_cma', help='Load latest CMA state or not',
                    action='store_true', default=False)

# initialize parameters
executable = None
exec_args = []
pre_value_args = []
exec_kwargs = {}
sleep_time = 10.0
wait_limit = 300.0
log_enabled = False
GPU_list = []
jobs_per_GPU = 0
env_seed = 0


def get_latest_subdir(parent_dir_path='.'):
    """
    Returns most recently created directory in given path
    """
    subdirs = []
    for subdir in os.listdir(parent_dir_path):
        subdirs.append(os.path.join(parent_dir_path, subdir))
    dir = max(subdirs, key=os.path.getmtime)
    return dir


def kill_carla():
    """
    Kill all CARLA processes
    """
    PROCNAME = "Carla"
    print('Killing existing CARLA servers...')
    for proc in psutil.process_iter():
        if PROCNAME in proc.name():
            pid = proc.pid
            os.kill(pid, 9)


def parse_unknown_args(args):
    """Parse arguments not consumed by arg parser into a dictionary."""
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def load(fi):
    """Load config variables from file."""
    try:
        # loading module via 'importlib', 'imp' deprecated with python3.5
        from importlib.machinery import SourceFileLoader
        m = SourceFileLoader('', str(fi)).load_module()
    except Exception:
        print('Could not load config file. Press any key to continue...')
        input()
        return
    global executable, exec_args, exec_kwargs, sleep_time, wait_limit, log_enabled, GPU_list, jobs_per_GPU, env_seed
    global pre_value_args

    if hasattr(m, 'executable'): executable = m.executable  # noqa
    if hasattr(m, 'exec_args'): exec_args = m.exec_args  # noqa
    if hasattr(m, 'pre_value_args'): pre_value_args = m.pre_value_args  # noqa
    if hasattr(m, 'exec_kwargs'): exec_kwargs.update(m.exec_kwargs)  # noqa
    if hasattr(m, 'sleep_time'): sleep_time = m.sleep_time  # noqa
    if hasattr(m, 'wait_limit'): wait_limit = m.wait_limit  # noqa
    if hasattr(m, 'log_enabled'): log_enabled = m.log_enabled  # noqa
    if hasattr(m, 'GPU_list'): GPU_list = m.GPU_list  # noqa
    if hasattr(m, 'jobs_per_GPU'): jobs_per_GPU = m.jobs_per_GPU  # noqa
    if hasattr(m, 'env_seed'): env_seed = m.env_seed  # noqa

# TODO are all jobs launched at once, meaning that finished jobs aren't immediately replaced?
def run_local_GPU(result_dir_path, valuation_file_path, solutions, gen, indiv_idx_array, retries=0):
    """
    Run individuals locally.
    """

    params_file = '%s/params_%d.npz' % (result_dir_path, gen)
    np.savez(params_file, params=solutions)

    # TODO some of these params should be set up top (or in a config file)
    ''' It is recommended to perform GPU arbitration via a centralized config file, below is an alternative solution,
    just remember to import nvidia-smi
    # # check if GPUs specified in config file are available
    # nvidia_smi.nvmlInit()
    # for GPU in GPU_list:
    #     res = nvidia_smi.nvmlDeviceGetUtilizationRates(nvidia_smi.nvmlDeviceGetHandleByIndex(GPU))
    #     print('GPU{}'.format(GPU))
    #     print(res.gpu)
    '''

    # jobs to run per gpu, number of gpus, number of jobs running per time
    NUM_GPU = len(GPU_list)
    num_jobs = NUM_GPU * jobs_per_GPU
    total_jobs = len(indiv_idx_array)

    active_procs_dict = dict.fromkeys(['GPU{}'.format(GPU_num) for GPU_num in GPU_list], [])
    for key in active_procs_dict:
        active_procs_dict['{}'.format(key)] = []

    jobs_start_idx = 0
    st_t = time.time()

    while jobs_start_idx < total_jobs: # TODO replace with while loop launching jobs dynamically
        jobs_failed = False
        launched_procs = []
        exit_codes = []

        # launching fixed number of jobs per iteration
        num_jobs = min(num_jobs, total_jobs - jobs_start_idx)
        print('num_jobs: {}'.format(num_jobs))
        print('jobs_start_idx: {}'.format(jobs_start_idx))
        batch_stt = time.time()

        for j in range(num_jobs):
            print('Start indiv_idx: {}, Current indiv_idx: {}'.format(jobs_start_idx, jobs_start_idx+j))
            indiv_idx = indiv_idx_array[jobs_start_idx + j]

            # set file path to save results in
            result_file_path = '%s/run_score.txt' % result_dir_path

            # create command-line string to run a job
            cmd = executable + ' '
            for val in pre_value_args:
                cmd += '%s ' % val
            cmd += ' %s ' % result_file_path
            cmd += ' %s ' % valuation_file_path
            cmd += ' %s ' % gen
            cmd += ' %s ' % indiv_idx
            for val in exec_args:
                cmd += '%s ' % val

            # specifying port, gpu number, seed
            cmd += '--params_file=%s ' % params_file
            cmd += '--gpu_num {} '.format(GPU_list[j // jobs_per_GPU]) # TODO assign to GPU based on availability of GPUs and constraints in config
            cmd += '--port {} '.format(7000 + j*10)
            cmd += '--seed {} '.format(env_seed)
            for key, val in exec_kwargs.items():
                cmd += '%s %s ' % (key, val)

            ## Spawn new process ##
            proc = subprocess.Popen(cmd.split())
            launched_procs.append(proc)
            active_procs_dict['GPU{}'.format(GPU_list[j // jobs_per_GPU])].append(proc.pid)

        print('----- WAITING FOR JOBS TO FINISH -----')

        try:
            exit_codes = [p.wait(timeout=36000) for p in launched_procs]
            print(exit_codes)

        except subprocess.TimeoutExpired:   # TODO have CARLA terminate runs before this would, so we can have a bad score included
            print('Killing process after timeout')
            print(active_procs_dict)
            jobs_failed = True
            with open(valuation_file_path, 'a+') as file:
                file.write('Killing process after timeout. Gen: {}\n'.format(gen))
                file.close()
            # p.kill()
            for liveproc in launched_procs:
                liveproc.kill()
                # os.killpg(os.getpgid(liveproc.pid), signal.SIGTERM)
            kill_carla()
            for key in active_procs_dict:
                active_procs_dict['{}'.format(key)] = []

        except any(exit_code != 0 for exit_code in exit_codes):
            print('Killing process after error')
            print(active_procs_dict)
            jobs_failed = True
            with open(valuation_file_path, 'a+') as file:
                file.write('Killing process after error. Gen: {} Exit_codes: {}\n'.format(gen, exit_codes))
                file.close()
            # p.kill()
            for liveproc in launched_procs:
                liveproc.kill()
                # os.killpg(os.getpgid(liveproc.pid), signal.SIGTERM)
            kill_carla()
            for key in active_procs_dict:
                active_procs_dict['{}'.format(key)] = []

        if not jobs_failed:
            jobs_start_idx += num_jobs

        batch_time = time.time() - batch_stt
        with open(valuation_file_path, 'a+') as file:
            file.write('Batch_time: {}\n\n'.format(batch_time))
            file.close()

        kill_carla()

    end_t = time.time()
    print('----- JOBS TERMINATED in {} -----'.format(end_t - st_t))
    # kill existing carla servers
    # above only kills the python training process, but still need to kill
    # carla server that was launched by that process (already done
    # in training code, but this here also as a safety measure)
    # kill any instance of CARLA before starting again
    kill_carla()


'''
# Generate samples / population for next evaluation.
def cma_es(experiment_path, gen, pop_size=0, in_file=None, seed=None):
    print(experiment_path, gen, pop_size, in_file, seed)
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    seed_path = in_file
    if not os.path.abspath(seed_path):
        seed_path = os.path.abspath(seed_path)
    os.chdir(os.path.join(current_dir, 'cma/'))
    if gen == 1:
        if in_file is None:
            raise ValueError('Must provide seed file.')
        if pop_size == 0:
            raise ValueError('Invalid population size.')
        # Launch for 1 iteration from beginning
        cmd = 'java -cp java cma.CMAMain %s %d %d %s'
        cmd = cmd % (experiment_path, gen, pop_size, seed_path)
    elif gen > 1:
        # Launch for 1 iteration from current
        cmd = 'java -cp java cma.CMAMain %s %d -c %d'
        cmd = cmd % (experiment_path, gen, gen - 1)
    else:
        raise ValueError('Invalid generation: %d' % gen)
    if seed is not None:
        cmd += ' --seed %d' % seed
    print(cmd)
    proc = subprocess.Popen(cmd.split(' '))
    proc.wait()
    os.chdir(current_dir)


def get_running_jobs(itr, pop_size, current_ids):
    """Get the IDs for running jobs."""
    out = subprocess.check_output('condor_q -nobatch'.split())
    if not isinstance(out, str):
        out = out.decode('utf8')
    out = out.split('\n')[4:-2]
    ids = []
    username = getpass.getuser()

    for line in out:
        if username in line:
            idx = line.split()[0][:-2]  # chop trailing .0
            if idx in current_ids:
                ids.append(idx)
    return ids


def get_remaining_trials(result_path, itr, pop_size):
    """Get inds where value files don't exit."""
    remaining_inds = []
    files = list(glob.iglob(os.path.join(result_path, 'run_%d_i_*.txt' % itr)))
    for i in range(pop_size):
        file = os.path.join(result_path, 'run_%d_i_%d.txt' % (itr, i))
        if file not in files:
            # if not os.path.exists(file):
            remaining_inds.append(i)
    return remaining_inds


def get_completed_trials(result_path, itr, pop_size):
    comp_inds = []
    files = list(glob.iglob(os.path.join(result_path, 'run_%d_i_*.txt' % itr)))
    for i in range(pop_size):
        file = os.path.join(result_path, 'run_%d_i_%d.txt' % (itr, i))
        if file in files:
            comp_inds.append(i)
    return comp_inds


def kill_remaining(job_ids):
    """Remove jobs submitted to condor."""
    print('Killing jobs:', job_ids)
    for idx in job_ids:
        try:
            subprocess.check_output(('condor_rm %s' % idx).split())
        except subprocess.CalledProcessError as e:
            print(e)


def get_failed_jobs(main_jobs, itr, pop_size):
    job_ids = [j[0] for j in main_jobs]
    running_jobs = get_running_jobs(itr, pop_size, job_ids)

    remaining_jobs = []
    for j in main_jobs:
        if j[0] not in running_jobs:
            remaining_jobs.append(j)

    return remaining_jobs


def remove_failed_jobs_from_list(main_jobs, failed_jobs):
    remaining_ids = [j[0] for j in failed_jobs]
    temp_running = []
    for j in main_jobs:
        if j[0] not in remaining_ids:
            temp_running.append(j)
    return temp_running
'''

def main():  # noqa

    global executable, exec_args, exec_kwargs, sleep_time, wait_limit, log_enabled, GPU_list, jobs_per_GPU, env_seed
    global pre_value_args

    # Argument usage information
    flags, unknown_args = parser.parse_known_args()
    
    ##
    # Set up file handling and load pre-trained model
    ##
    last_experiment_path = get_latest_subdir(flags.experiment_path)
    experiment_path = os.path.join(flags.experiment_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    process_dir_path = os.path.join(experiment_path, 'process')
    result_dir_path = os.path.join(experiment_path, 'results')
    valuation_file_path = os.path.join(result_dir_path, 'valuationdone.txt')

    # Load config variables
    load(flags.config_file)
    if flags.executable is not None:
        print('Overriding executable in config file.')
        executable = flags.executable
    print('unknown args:{}'.format(unknown_args))
    unknown_args = parse_unknown_args(unknown_args)
    exec_kwargs.update(unknown_args)

    # Init directory
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    if not os.path.exists(process_dir_path):
        os.mkdir(process_dir_path)
    if os.path.exists(result_dir_path):
        shutil.rmtree(result_dir_path)
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)

    # loading params of pre-trained model to tune (already saved in file beforehand)
    model_params = []
    with open('/home/boschaustin/projects/CL_AD/ES/carla_lbc_new/training/model_seed_params.txt', 'r') as f:
        line = f.readline()
        while line:
            model_params.append(float(line))
            line = f.readline()


    ##
    # Start CMA-ES process in a never-ending loop
    ##

    if flags.load_cma:
        cma_state_file_path = os.path.join(last_experiment_path, 'results/cma-state')
        es = pickle.load(open(cma_state_file_path, 'rb'))
        print('Resuming from cma-state file in:{}\n'.format(cma_state_file_path))
        with open(valuation_file_path, 'a+') as file:
            file.write('Resuming from cma-state file in:{}\n'.format(cma_state_file_path))
            file.close()
    else:
        # create CMAES object and variance to perturb each parameter
        es = cma.CMAEvolutionStrategy(model_params, 0.0005)     # TODO check the appropriateness of this variance value

    itr = 0
    while not es.stop():  # 1 loop per generation

        # Generate population
        if flags.pop_size == -1:
            solutions = es.ask()
        else:
            solutions = es.ask(flags.pop_size)
        pop_size = len(solutions)
        indiv_array = [i for i in range(pop_size)]

        # Evaluate population
        if flags.run_local:
            start_time = time.time()
            run_local_GPU(result_dir_path, valuation_file_path, solutions, itr, indiv_array)
            gen_eval_time = time.time()-start_time
        else:
            print('failed to parse flag --run_local')
            break

        src = os.path.join(result_dir_path, 'run_score.txt')
        dest = os.path.join(result_dir_path, 'value_score.txt')

        # Build dataframe from result file with columns: Gen, Indiv, Fit
        result_df = pd.read_csv(src, sep=' ', header=None, names=['Gen', 'Indiv', 'Fit'])

        # note: taking negative of fitness score since cma library
        # minimizes objective
        result_df['Fit'] = result_df['Fit'].apply(lambda x: x*-1)

        # Reduce dataframe to results of current generation
        result_df = result_df[result_df['Gen'] == itr]

        # Create list of values from Fit
        values = result_df['Fit'].values.tolist()

        # Re-order solutions according to Indiv evaluations (and grab only solutions that produced a result)
        temp_solutions = [solutions[idx] for idx in result_df['Indiv'].values.tolist()]

        with open(src):
            try:
                shutil.copyfile(src, dest)
            except Exception as e:
                raise e

        print('Generation evaluation complete. Number of individuals evaluated:', len(temp_solutions))
        es.tell(temp_solutions, values)  # send results to CMA-ES, to be used for sampling the next population
        filename = os.path.join(result_dir_path, 'cma-state')
        open(filename, 'wb').write(es.pickle_dumps())
        es.logger.add()

        with open(valuation_file_path, 'a+') as file:
            file.write('Jobs_GPU %d Num_GPU %d Gen %d Eval_time %d\n' % (jobs_per_GPU, len(GPU_list), itr, gen_eval_time))
            file.close()

        itr += 1
        print('finished iteration {}'.format(itr))

    print('CMA-ES finished')


if __name__ == '__main__':
    main()

