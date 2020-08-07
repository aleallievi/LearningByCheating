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

import glob
import sys
import torch
import subprocess
import signal

try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
    sys.path.append(glob.glob('../')[0])
except IndexError as e:
    pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from bird_view.utils.traffic_events import TrafficEventType

from bird_view.utils import carla_utils as cu
from benchmark import make_suite

BACKBONE = 'resnet34'
GAP = 5
N_STEP = 5
CROP_SIZE = 192
MAP_SIZE = 320
SAVE_EPISODES = list(range(20))

def rollout(replay_buffer, net,  
        image_agent_kwargs=dict(), episode_length=1000,
        n_vehicles=100, n_pedestrians=250, port=2000, planner="new"):

    from models.image import ImageAgent
    
    num_data = 0

    weathers = list(cu.TRAIN_WEATHERS.keys())
    fit = 0
    tot_t = 0
    for weather in weathers:
    
        data = list()

        with make_suite('FullTown01-v1', port=port, planner=planner) as env:
            start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
            env_params = {
                'weather': weather,
                'start': start,
                'target': target,
                'n_pedestrians': n_pedestrians,
                'n_vehicles': n_vehicles,
                }

            env.init(**env_params)
            dist_to_cover = env.get_distance_start_to_goal()

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
            while True:
                if env.is_success() or time_step >= episode_length:
                    break

                env.tick()
                coll_actors = env.collision_actors
                route_comp = env.route_completed
                red_count = env.red_count
                stop_count = env.stop_count              
                
                if time_step % 100 == 0:
                    print ('weather {}'.format(weather))
                    print ('time steps so far {}'.format(time_step))
                    print ('route completed so far {}'.format(route_comp))
                    print ('coll actors so far {}'.format(coll_actors))
                    print ('red count so far {}'.format(red_count))
                    print ('stop count so far {}'.format(stop_count))  
                
                observations = env.get_observations()
                control = image_agent.run_step(observations) 
                diagnostic = env.apply_control(control)
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
            
            infraction_pen = stop_pen * red_pen * (coll_stat_pen * coll_veh_pen * coll_ped_pen)
            score = route_comp * infraction_pen
            end_t = time.time()
            print ('========= Episode Stats (Weather {}) ========='.format(weather))
            print ('Total time: {}'.format(time_step))
            print ('Route Completion Percentage: {}'.format(route_comp))
            print ('Collided into: {}'.format(coll_actors))
            print ('Number of Red Lights run: {}'.format(red_count))
            print ('Number of Stop Signs run: {}'.format(stop_count))            
            print ('Driving Score: {}'.format(score))
            print ('Epsiode Time (real-time, seconds): {}'.format(end_t - st_t))
            print ('=================================')
            fit += score
            tot_t += (end_t - st_t)
    
    print ('Total Fitness: {}'.format(fit))
    print ('Total Time (s): {}'.format(tot_t))
    return fit

def launch_carla(port):
    cmd = '/home/boschaustin/projects/CL_AD/ES/carla_lbc/CarlaUE4.sh -fps=10 -no-rendering -carla-world-port={}'.format(port)
    carla_process = subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
    # give few seconds to bootup
    time.sleep(5)
    return carla_process

def train(output_file, params_file, cur_ind, model_path, config):

    from phase2_utils import (
        ReplayBuffer, 
        load_image_model,
        )
    
    net = load_image_model(
        config['model_args']['backbone'], 
        model_path,
        device=config['device'])

    params = np.load(params_file)['params'][cur_ind]

    di = net.state_dict()
    st = 0
    for i in range(4):

        w_key = 'location_pred.{}.1.weight'.format(i)
        b_key = 'location_pred.{}.1.bias'.format(i)

        w_shape = di[w_key].shape
        w_num = np.prod(w_shape)
        b_shape = di[b_key].shape
        b_num = np.prod(b_shape)

        w = params[st : st + w_num]
        b = params[st + w_num : st + w_num + b_num]
        st += (w_num + b_num)

        reshaped_w = np.reshape(w, w_shape)
        
        di[w_key] = torch.from_numpy(reshaped_w)
        di[b_key] = torch.from_numpy(b)
    net.load_state_dict(di)
    
    image_agent_kwargs = { 'camera_args' : config["agent_args"]['camera_args'] }

    replay_buffer = ReplayBuffer(**config["buffer_args"])
    
    ret = rollout(replay_buffer, net, image_agent_kwargs=image_agent_kwargs, port=config['port'])

    all_dists = []
    total_fit = ret
    all_dists.append(total_fit)
    avg_dist = np.mean(all_dists)

    print ('Writing average distance..')
    with open(output_file, 'w+') as w:
        w.write(str(float(total_fit)))
    with open('%s.rew' % output_file, 'w') as w:
        for dist in all_dists:
            w.write('%f\n' % dist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Optimize simulator trajectories to match reference trajectory.')
    parser.add_argument('result_file', type=str, help='File to write results to.')
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
    parsed = parser.parse_args()

    output_file = parsed.result_file
    ind = int(output_file.split('_')[-1].split('.')[0])
    model_path = parsed.model_path
    params_file = parsed.params_file

    config = {
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


    PROCNAME = "Carla"

    for proc in psutil.process_iter():
        if PROCNAME in proc.name():
            pid = proc.pid
            #os.kill(pid, 9)
    # launch carla server
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(parsed.gpu_num)
    
    carla_process = launch_carla(port = config['port'])
    print ('Launching carla on GPU {} and port {}'.format(parsed.gpu_num, parsed.port))
    
    st = time.time()
    train(output_file,
            params_file,
            ind,
            model_path,
            config)
    print ('Total time for all weathers {}'.format(time.time() - st))

    # kill carla
    os.killpg(os.getpgid(carla_process.pid), signal.SIGTERM)
