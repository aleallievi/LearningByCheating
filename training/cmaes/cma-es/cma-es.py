#!/usr/bin/env python
"""Run CMA-ES on generic task."""

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

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('experiment_path', type=str,
                    help='Directory for results.')
parser.add_argument('--executable', default=None, type=str, help='Path to executable.')
#parser.add_argument('num_iters', help='Number of iterations.',type=int)
parser.add_argument('--pop_size', help='Population size.',type=int, default=-1)
parser.add_argument('config_file', type=str, help='Configuration variables.')
#parser.add_argument('--seed_file', help='Seed file for CMA-ES.', default=None)
#parser.add_argument('--start_iter', help='Iteration to start from.', type=int,default=1)
parser.add_argument('--run_local', help='Launch on condor or not',
                    action='store_true', default=False)

# flags, unknown_flags = parser.parse_known_args()

executable = None
exec_args = []
pre_value_args = []
exec_kwargs = {}
sleep_time = 10.0
wait_limit = 300.0
log_enabled = False


def parse_unknown_args(args):
    """Parse arguments not consumed by arg parser into a dicitonary."""
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
        import imp
        with open(fi) as f:
            m = imp.load_source('m', '', f)
    except Exception:
        print('Could not load config file. Continuing.')
        return
    global executable, exec_args, exec_kwargs, sleep_time, wait_limit, log_enabled
    global pre_value_args
    if hasattr(m, 'executable'): executable = m.executable  # noqa
    if hasattr(m, 'exec_args'): exec_args = m.exec_args  # noqa
    if hasattr(m, 'pre_value_args'): pre_value_args = m.pre_value_args  # noqa
    if hasattr(m, 'exec_kwargs'): exec_kwargs.update(m.exec_kwargs)  #= m.exec_kwargs  # noqa
    if hasattr(m, 'sleep_time'): sleep_time = m.sleep_time  # noqa
    if hasattr(m, 'wait_limit'): wait_limit = m.wait_limit  # noqa
    if hasattr(m, 'log_enabled'): log_enabled = m.log_enabled  # noqa

# TODO are all jobs launched at once, meaning that finished jobs aren't immediately replaced?
def run_local_GPU(experiment_path, solutions, gen, inds, seeds, retries=0):
    """Run individuals locally."""
    params_file = '%s/results/params_%d.npz' % (experiment_path, gen)
    np.savez(params_file, params=solutions)
    
    # TODO some of these params should be set up top (or in a config file)
    env_seed = 0 # using 0 seed, the most common one used by LbC; same seed creates less variation in evaluation; note either CARLA or the LbC model have their own stochasticity, independent of this seed TODO use the same seed for the policy if it's probabilistic
    # jobs to run per gpu, number of gpus, number of jobs running per time
    jobs_per_gpu = 2
    NUM_GPU = 4
    num_jobs = NUM_GPU * jobs_per_gpu
    num_launched = 0
    #for ind, seed in zip(inds, seeds):
    st_t = time.time()
    for i in range(0, len(inds), num_jobs):
        launched_procs = []
        # launching fixed number of jobs per time
        for j in range(num_jobs):
            ind = inds[i + j]
            # note this seed is diferent from seed used to generate environment 
            seed = seeds[i + j] # TODO see if no errors after this is deleted
            #params_file = '%s/results/params_%d_i_%d.txt' % (experiment_path, gen, ind)
            
            # set file path to save results in
            value_file = '%s/results/run_%d_i_%d.txt' % (experiment_path, gen, ind) # TODO if there's only one line of text per file, why not just append all results to the same file?
            
            # create command-line string to run a job 
            cmd = executable + ' '
            for val in pre_value_args:
                cmd += '%s ' % val
            cmd += ' %s ' % value_file
            for val in exec_args:
                cmd += '%s ' % val

            # specifying port, gpu number, seed
            cmd += '--params_file=%s ' % params_file
            cmd += '--gpu_num {} '.format(j // jobs_per_gpu) # TODO assign to GPU based on availability of GPUs and constraints in config
            cmd += '--port {} '.format((j + 2) * 1000)
            cmd += '--seed {} '.format(env_seed)
            for key, val in exec_kwargs.items():
                cmd += '%s %s ' % (key, val)
            
            ## Spawn new process ##
            print(cmd)
            # proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            proc = subprocess.Popen(cmd.split())
            #proc.wait()
            launched_procs.append(proc)

        print ('----- WAITING FOR JOBS TO FINISH -----')
        for p in launched_procs:
            try:
                # wait for an hr
                p.wait(timeout = 3600) # TODO have CARLA terminate runs before this would, so we can have a bad score included
            except TimeoutExpired:
                print('Killing process after timeout. Gen', gen, ', indiv', ind)  
                p.kill()
        end_t = time.time()
        print ('----- JOBS TERMINATED in {} -----'.format(end_t - st_t))
        #time.sleep(jobs_per_gpu * 1200)
        # kill existing carla servers
        PROCNAME = "Carla"

        # above only kills the python training process, but still need to kill
        # carla server that was launched by that process (already done
        # in training code, but this here also as a safety measure)
        # kill any instance of CARLA before starting again
        for proc in psutil.process_iter():
            if PROCNAME in proc.name():
                pid = proc.pid
                os.kill(pid, 9)

        #print ('break point')
        #pdb.set_trace()

# NOT USED
'''
def run_local(experiment_path, solutions, gen, inds, seeds, retries=0):
    """Run individuals locally."""
    params_file = '%s/results/params_%d.npz' % (experiment_path, gen)
    np.savez(params_file, params=solutions)
    for ind, seed in zip(inds, seeds):
        #params_file = '%s/results/params_%d_i_%d.txt' % (experiment_path, gen, ind)
        value_file = '%s/results/run_%d_i_%d.txt' % (experiment_path, gen, ind)
        cmd = executable + ' '
        for val in pre_value_args:
            cmd += '%s ' % val
        cmd += ' %s ' % value_file
        for val in exec_args:
            cmd += '%s ' % val
        cmd += '--params_file=%s ' % params_file
        cmd += '--gpu_num 3 '
        for key, val in exec_kwargs.items():
            cmd += '%s %s ' % (key, val)
        print(cmd)
        # proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        proc = subprocess.Popen(cmd.split())
        proc.wait()
'''

'''
#Requirements = ARCH == "X86_64" && !GPU
#Requirements=InPublic
#Requirements = ARCH == "X86_64" && !GPU
def run_on_condor(experiment_path, solutions, gen, inds, seeds, retries=0):
    #Launch evaluations for given generation and population members.
    params_file = '%s/results/params_%d.npz' % (experiment_path, gen)
    np.savez(params_file, params=solutions)
    ids = []
    for ind, seed in zip(inds, seeds):
        #params_file = '%s/results/params_%d_i_%d.txt' % (experiment_path, gen, ind)
        value_file = '%s/results/run_%d_i_%d.txt' % (experiment_path, gen, ind)
        condor_contents = """Executable = %s
Universe = vanilla
Environment = ONCONDOR=true
Getenv = true
+GPUJob = true
Requirements=(TARGET.GPUSlot) 
Rank = -SlotId + !InMastodon*10

+Group = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "CMA-ES Experiments"

Input = /dev/null
"""  % executable
        if log_enabled:
            condor_contents += 'Error = %s/process/error-%d-%d-%d.err\n' % (experiment_path, gen, ind, retries)  # noqa
            condor_contents += 'Output = %s/process/out-%d-%d-%d.out\n' % (experiment_path, gen, ind, retries)  # noqa
            condor_contents += 'Log = %s/process/log-%d-%d-%d.log\n' % (experiment_path, gen, ind, retries)  # noqa
        else:
            condor_contents += 'Error = /dev/null\n'
            condor_contents += 'Output = /dev/null\n'
            condor_contents += 'Log = /dev/null\n'
        condor_contents += 'arguments = '

        for val in pre_value_args:
            condor_contents += '%s ' % val
        condor_contents += '%s ' % value_file

        for val in exec_args:
            condor_contents += '%s ' % val
        condor_contents += ' --params_file=%s ' % params_file
        for key, val in exec_kwargs.items():
            condor_contents += '%s %s ' % (key, val)
        condor_contents += '\nQueue 1'
        # print(condor_contents)
        # raise NotImplementedError
        # Submit Job
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(condor_contents.encode())
        proc.stdin.close()
        proc.wait()
        for line in proc.stdout:
            if not isinstance(line, str):
                line = line.decode('utf8')
            if 'cluster' in line:
                #ids.append(line.split()[-1][:-1])
                ids.append((line.split()[-1][:-1], ind))
        time.sleep(0.05)
    print('Submitted %d jobs' % len(inds))
    return ids
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


#def get_running_jobs(result_path, itr, pop_size, current_ids):
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




def main():  # noqa

    global executable, exec_args, exec_kwargs, sleep_time, wait_limit, log_enabled
    global pre_value_args

    # Argument usage information
    flags, unknown_args = parser.parse_known_args()
    
    ##
    # Set up file handling and load pre-trained model
    ##

    global executable # TODO redundant?
    experiment_path = flags.experiment_path
    process_path = os.path.join(experiment_path, 'process')
    result_path = os.path.join(experiment_path, 'results')
    # Load config variables
    load(flags.config_file)
    if flags.executable is not None:
        print('Overriding executable in config file.')
        executable = flags.executable
    print(unknown_args)
    unknown_args = parse_unknown_args(unknown_args)
    exec_kwargs.update(unknown_args)

    # Init directory
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    if not os.path.exists(process_path):
        os.mkdir(process_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # loading params of pre-trained model to tune (already saved in file beforehand)
    model_params = []
    with open('/home/boschaustin/projects/CL_AD/ES/carla_lbc/training/model_seed_params.txt', 'r') as f:
        line = f.readline()
        while line:
            model_params.append(float(line))
            line = f.readline()


    ##
    # Start CMA-ES process in a never-ending loop
    ##

    # create CMAES object and variance to perturb each parameter
    es = cma.CMAEvolutionStrategy(model_params, 0.0005) # TODO check the appropriateness of this variance value
    
    #for itr in range(start_iter, n_iters + 1):  # CMA-ES code requires 1-indexing
    itr = 0
    while not es.stop(): # 1 loop per generation
        
        # Generate population
        if flags.pop_size == -1:
            solutions = es.ask() # TODO change "solutions" variable name to "drawn_params"
        else:
            solutions = es.ask(flags.pop_size)
        pop_size = len(solutions)
        inds = [i for i in range(pop_size)]
        seeds = np.random.randint(0.4e10, size=pop_size) # TODO Is this used?

        # Evaluate population
        if flags.run_local:
            run_local_GPU(experiment_path, solutions, itr, inds, seeds)
        else:
            # NOTE: no quality assurance on the else branch here. Was used for condor
            retry = 1
            main_jobs = run_on_condor(experiment_path, solutions, itr, inds, seeds, retries=0)
            time.sleep(sleep_time)
            failed_jobs = get_failed_jobs(main_jobs, itr, pop_size)
            completed_inds = []
            while True:
                if len(failed_jobs) > 0:
                    # update main jobs list
                    main_jobs = remove_failed_jobs_from_list(main_jobs, failed_jobs)
                    # relaunch failed pop members
                    remaining_inds = [j[1] for j in failed_jobs]
                    if len(completed_inds) > 0:
                        temp_rem_inds = []
                        for j in remaining_inds:
                            if j not in completed_inds:
                                temp_rem_inds.append(j)
                        remaining_inds = temp_rem_inds
                    print ('number of failed jobs {}'.format(len(remaining_inds)))
                    relaunched_jobs = run_on_condor(experiment_path, solutions, itr, remaining_inds, seeds[remaining_inds], retries=retry)
                    retry += 1
                    # add newly launched jobs to main running list
                    for rj in relaunched_jobs:
                        main_jobs.append(rj)
                    assert len(main_jobs)  - len(completed_inds) == pop_size

                print ('number of completed jobs {}'.format(len(completed_inds)))
                print('Waiting for %d unfinished jobs. Sleeping for %ds' % (len(main_jobs),
                                                                            sleep_time))
                time.sleep(sleep_time)
                failed_jobs = get_failed_jobs(main_jobs, itr, pop_size)

                completed_inds = get_completed_trials(result_path, itr, pop_size)
                incomplete_inds = get_remaining_trials(result_path, itr, pop_size)
                if len(incomplete_inds) <= 0.1 * pop_size:
                    break

            '''
            remaining_inds = get_remaining_trials(result_path, itr, pop_size)
            job_ids = run_on_condor(experiment_path, solutions, itr, inds, seeds, retries=0)
            # Wait for completion
            wait_ct = 0
            retry = 1
            while len(remaining_inds) > 0:
                if wait_ct > wait_limit or (not job_ids and remaining_inds):
                    # If no jobs are running but some files aren't written then
                    # the jobs may have failed. We first pause to make sure it
                    # isn't just a delay in writing the results.
                    if not job_ids and remaining_inds:
                        time.sleep(5.0)
                        print ('checking result path {}'.format(result_path))
                        remaining_inds = get_remaining_trials(result_path, itr, pop_size)
                        break
                        if not remaining_inds:
                            break
                        elif len(remaining_inds) < 0.1 * pop_size:
                            print('Not all jobs complete but missing < 10%. Continuing.')
                            break
                        print('Jobs failed.')
                    else:
                        print('Jobs took too long.')
                    print('Killing and resubmitting.')
                    if job_ids:
                        kill_remaining(job_ids)
                    job_ids = run_on_condor(experiment_path, solutions, itr, remaining_inds,
                                            seeds[remaining_inds], retries=retry)
                    wait_ct = 0
                    retry += 1
                print('Waiting for %d unfinished jobs. Sleeping for %ds' % (len(job_ids),
                                                                            sleep_time))
                time.sleep(sleep_time)
                wait_ct += sleep_time
                remaining_inds = get_remaining_trials(result_path, itr, pop_size)
                job_ids = get_running_jobs(result_path, itr, pop_size, job_ids)

                if len(remaining_inds)==0 and len(job_ids)>0:
                    time.sleep(10)
                    job_ids = get_running_jobs(result_path, itr, pop_size, job_ids)
                '''

            # kill_remaining(job_ids)

        temp_sol=[]
        values = []
        for ind in range(pop_size):
            src = os.path.join(result_path, 'run_%d_i_%d.txt' % (itr, ind))
            dest = os.path.join(result_path, 'value_%d_i_%d.txt' % (itr, ind))
            if os.path.exists(src):
                # note: taking negative of fitness score since cma library
                # minimizes objective
                with open(src) as f:
                    values.append(-float(f.readline()))
                temp_sol.append(solutions[ind])
                try:
                    shutil.move(src, dest)
                except Exception as e:
                    raise e

        #es.tell(solutions, values)
        es.tell(temp_sol, values) # send results to CMA-ES, to be used for sampling the next population
        filename = 'cma-state'
        open(filename, 'wb').write(es.pickle_dumps())
        es.logger.add()


        with open(os.path.join(result_path, 'valuationdone_%d.txt' % itr), 'w'):
            print('Evaluation done for generation %d' % itr)

        itr += 1
        print ('finished iteration {}'.format(itr))

    print ('CMA-ES finished')

if __name__ == '__main__':
    main()



