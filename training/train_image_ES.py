#!/usr/bin/env python
import os
import psutil
import argparse
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
import pygame
import glob
import sys
import torch
import subprocess
import signal
import pandas as pd

# add paths needed to import required modules to sys.path
path = Path(os.getcwd())
try:
    sys.path.append(glob.glob(str(path.parent) + '/bird_view')[0])
    sys.path.append(glob.glob(str(path.parent))[0])
    sys.path.append(glob.glob(str(path.parent) + '/PythonAPI')[0])
except IndexError as e:
    pass

# import additional modules
from bird_view.utils.traffic_events import TrafficEventType
from bird_view.utils import carla_utils as cu
from benchmark import make_suite, _suites
from benchmark.goal_suite import from_file


def str2bool(v):
    """
    This function accepts a string and returns its boolean value.
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def env_rollout(weather, start, target, n_pedestrians, n_vehicles, net,
                image_agent_kwargs, ImageAgent, suite_name, port, planner, env_seed,
                valuation_file_path, indiv_idx, episode_length):
    # Initialize fitness score and execution_time variables
    fit = 0
    tot_t = 0
    # Instantiate environment "env" through make_suite (in benchmark.__init__.py)
    # Note: make_suite provides a wrapper to launch and access the CARLA client and server;
    # magic functions, performs __init__, __enter__, __exit__ are executed in this order inside the "with" loop
    with make_suite(suite_name, port=port, planner=planner, env_seed=env_seed) as env:
        # Set environment parameters and initialize environment
        env_params = {
            'weather': weather,
            'start': start,
            'target': target,
            'n_pedestrians': n_pedestrians,
            'n_vehicles': n_vehicles,
        }
        # env.seed = env_seed
        # Initialize environment
        env.init(**env_params)
        # success distance threshold from target
        env.success_dist = 10.0

        image_agent_kwargs['model'] = net
        image_agent = ImageAgent(**image_agent_kwargs)

        coll_actors = None
        route_comp = 0
        red_count = -1
        stop_count = -1

        time_step = 0
        # TODO: just using (https://carlachallenge.org/challenge/) as temp sol
        # i.e. terminating only if critical infraction as defined above
        # also keeping a timeout episode_length
        st_t = time.time()
        while not env.is_success() and not time_step >= episode_length:
            env.tick()
            coll_actors = env.collision_actors
            route_comp = env.route_completed
            red_count = env.red_count
            stop_count = env.stop_count

            # if time_step % 100 == 0:
            #     print('weather {}'.format(weather))
            #     print('time steps so far {}'.format(time_step))
            #     print('route completed so far {}'.format(route_comp))
            #     print('coll actors so far {}'.format(coll_actors))
            #     print('red count so far {}'.format(red_count))
            #     print('stop count so far {}'.format(stop_count))

            observations = env.get_observations()
            control = image_agent.run_step(observations)
            env.apply_control(control)
            time_step += 1

        stop_pen = 0.8 ** stop_count
        red_pen = 0.7 ** red_count
        coll_stat_pen = 1.
        coll_veh_pen = 1.
        coll_ped_pen = 1.
        for c in coll_actors:
            if c == TrafficEventType.COLLISION_STATIC:
                coll_stat_pen *= 0.65
            elif c == TrafficEventType.COLLISION_VEHICLE:
                coll_veh_pen *= 0.6
            elif c == TrafficEventType.COLLISION_PEDESTRIAN:
                coll_ped_pen *= 0.5

        # computing driving score of episode
        infraction_pen = stop_pen * red_pen * (coll_stat_pen * coll_veh_pen * coll_ped_pen)
        score = route_comp * infraction_pen
        end_t = time.time()
        print('========= Episode Stats (Weather {}) ========='.format(weather))
        print('Total time: {}'.format(time_step))
        print('Route Completion Percentage: {}'.format(route_comp))
        print('Collided into: {}'.format(coll_actors))
        print('Number of Red Lights run: {}'.format(red_count))
        print('Number of Stop Signs run: {}'.format(stop_count))
        print('Driving Score: {}'.format(score))
        print('Episode Time (real-time, seconds): {}'.format(end_t - st_t))
        print('=================================')

        # Save weather valuation details to valuation file
        with open(valuation_file_path, 'a+') as file:
            file.write(
                'Indiv:{} Weather:{} Total_time:{} Route_Comp:{} Collisions:{} Red_lights:{} Stop_signs:{} Score:{}'
                ' Ep_time:{} St_t:{} End_t:{} Start-Target_poses:{}\n'.format(
                    indiv_idx, weather, time_step, route_comp, coll_actors, red_count, stop_count, score,
                    end_t - st_t, st_t, end_t, (start, target)))
            file.close()
        fit += score
        tot_t += (end_t - st_t)
        env.terminate_criterion_test()
    return fit, tot_t


def rollout(net, valuation_file_path, indiv_idx, image_agent_kwargs=dict(),
            episode_length=1000, n_vehicles=100, n_pedestrians=250, port=2000, planner="new", env_seed=2020):
    """This function performs episode evaluations based on the specified parameters."""

    # Import agent model
    from models.image import ImageAgent

    # Specify list of weathers
    weathers = list(sorted(cu.TRAIN_WEATHERS.keys()))

    # Initialize fitness score and execution_time variables
    fit = 0
    tot_t = 0

    # Get start and target pose tasks TODO: refactor to avoid computing pose_tasks again when "env" is instantiated
    suite_name = 'FullTown01-v1'
    args, kwargs = _suites[suite_name]
    pose_tasks = from_file(kwargs['poses_txt'])[0:2]

    # Weather evaluation loop
    for weather in weathers:
        # Start, target evaluation loop
        for start, target in [pose_tasks[i] for i in range(len(pose_tasks))]:
            target_stt = time.time()
            fit, tot_t = env_rollout(weather, start, target, n_pedestrians, n_vehicles, net, image_agent_kwargs,
                                     ImageAgent, suite_name, port, planner, env_seed, valuation_file_path,
                                     indiv_idx, episode_length)

    print('Total Fitness: {}'.format(fit))
    print('Total Time (s): {}'.format(tot_t))
    return fit


def launch_carla(port):
    """Carla server instantiation"""
    # launches CarlaUE4.sh as subprocess
    cmd = '/home/boschaustin/projects/CL_AD/ES/carla_lbc_new/CarlaUE4.sh ' \
          '-fps=10 -no-rendering -carla-world-port={}'.format(port)
    carla_process = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
    # give a few seconds to boot-up
    time.sleep(5)
    return carla_process


def train(result_file_path, valuation_file_path, params_file, gen, indiv_idx, model_path, config, carla_process):
    from phase2_utils import (
        ReplayBuffer, 
        load_image_model,
        )

    # load pre-trained model
    net = load_image_model(
        config['model_args']['backbone'], 
        model_path,
        device=config['device'])

    # load CMAES parameters, indiv_idx is the population member id i.e. [0, pop_size-1]
    params = np.load(params_file)['params'][indiv_idx]

    # overwrite weights
    di = net.state_dict()
    st = 0
    for i in range(4):

        w_key = 'location_pred.{}.1.weight'.format(i)
        b_key = 'location_pred.{}.1.bias'.format(i)

        w_shape = di[w_key].shape
        w_num = np.prod(w_shape)
        b_shape = di[b_key].shape
        b_num = np.prod(b_shape)

        w = params[st: st + w_num]
        b = params[st + w_num: st + w_num + b_num]
        st += (w_num + b_num)

        reshaped_w = np.reshape(w, w_shape)
        
        di[w_key] = torch.from_numpy(reshaped_w)
        di[b_key] = torch.from_numpy(b)
    net.load_state_dict(di)
    
    image_agent_kwargs = {'camera_args': config["agent_args"]['camera_args']}

    ret = rollout(net, valuation_file_path, indiv_idx, episode_length=1000, image_agent_kwargs=image_agent_kwargs,
                  port=config['port'], env_seed=config['env_seed'])

    all_dists = []
    total_fit = ret
    all_dists.append(total_fit)
    avg_dist = np.mean(all_dists)

    # write to results files by appending new line using format: generation individual_ID score \n
    print('Writing to results files..')
    with open(result_file_path, 'a+') as file:
        file.write(str(int(gen)) + ' ' + str(int(indiv_idx)) + ' ' + str(float(total_fit)) + '\n')
        file.close()
    with open('%s.rew' % result_file_path, 'a+') as file:
        for dist in all_dists:
            file.write('%f %f %f\n' % (gen, indiv_idx, dist))
            file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Optimize simulator trajectories to match reference trajectory.')
    parser.add_argument('result_file_path', type=str, help='File to write results to.')
    parser.add_argument('valuation_file_path', type=str, help='File to write valuation details to.')
    parser.add_argument('gen', type=int, help='Generation id.')
    parser.add_argument('indiv_idx', type=int, help='Individual id.')
    parser.add_argument('--model_path', help='pre-trained model', type=str)
    parser.add_argument('--params_file', help='Simulator parameters.', type=str, default=None)
    parser.add_argument('--gpu_num', default='0')
    parser.add_argument('--log_dir', required=False)
    parser.add_argument('--log_iterations', default=100)
    parser.add_argument('--max_episode', default=20)
    parser.add_argument('--epoch_per_episode', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--speed_noise', type=float, default=0.0)
    parser.add_argument('--batch_aug', type=int, default=1)

    parser.add_argument('--ckpt', required=False)

    parser.add_argument('--fixed_offset', type=float, default=4.0)

    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)

    # Misc
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2020)
    parsed = parser.parse_args()

    result_file_path = parsed.result_file_path
    valuation_file_path = parsed.valuation_file_path
    indiv_idx = parsed.indiv_idx
    gen = parsed.gen
    model_path = parsed.model_path
    params_file = parsed.params_file

    BACKBONE = 'resnet34'
    GAP = 5
    N_STEP = 5
    CROP_SIZE = 192
    MAP_SIZE = 320
    SAVE_EPISODES = list(range(20))

    # specify port to launch server on and connect agent to
    # specify which device to use CPU or GPU etc
    config = {
            'env_seed': parsed.seed,
            'port': parsed.port,
            'log_dir': parsed.log_dir,
            'log_iterations': parsed.log_iterations,
            'batch_size': parsed.batch_size,
            'max_episode': parsed.max_episode,
            'speed_noise': parsed.speed_noise,
            'epoch_per_episode': parsed.epoch_per_episode,
            #'device': 'cpu',
            'device': 'cuda',
            'phase1_ckpt': parsed.ckpt,
            'optimizer_args': {'lr': parsed.lr},
            'buffer_args': {
                'buffer_limit': 200000,
                'batch_aug': parsed.batch_aug,
                'augment': 'super_hard',
                'aug_fix_iter': 819200,
            },
            'model_args': {
                'model': 'image_ss',
                'backbone': BACKBONE,
                },
            'agent_args': {
                'camera_args': {
                    'w': 384,
                    'h': 160,
                    'fov': 90,
                    'world_y': 1.4,
                    'fixed_offset': parsed.fixed_offset,
                }
            },
        }

    # environment variable to set to allocate this job to a specific GPU
    # for example, what to set to 2nd GPU, the parsed.gpu_num = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(parsed.gpu_num)

    # launch carla server
    carla_process = launch_carla(port=config['port'])

    st = time.time()
    train(result_file_path,
          valuation_file_path,
          params_file,
          gen,
          indiv_idx,
          model_path,
          config,
          carla_process)

    print('Total time for all weathers and targets {}'.format(time.time() - st))
    with open(valuation_file_path, 'a+') as file:
        file.write('Total time for all weathers and targets {}'.format(time.time() - st))
        file.close()

    # kill carla
    # os.killpg(os.getpgid(carla_process.pid), signal.SIGTERM)

    # print('Carla process to kill on GPU {} and port {}'.format(parsed.gpu_num, parsed.port))
    # print('Carla pid to kill is: {}'.format(os.getpgid(carla_process.pid)))
    # p = subprocess.Popen(["ps -o cmd= {}".format(carla_process.pid)], stdout=subprocess.PIPE, shell=True)
    # print(str(p.communicate()[0]))

    # Kill only parent process (not working either)
    # p = subprocess.Popen(['pgrep -g {}'.format(os.getpgid(carla_process.pid))], stdout=subprocess.PIPE, shell=True)
    # print(carla_process.pid)
    # out = str(p.communicate()[0])
    # ppid = int(out.split('\\n')[1])
    # os.kill(ppid, signal.SIGTERM)
    # print('*********************************')
