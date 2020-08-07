import numpy as np

import gym
from gym import spaces
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class GymSim(object):
    def __init__(self, env, seed, visual=False, stack_size=3):
        self.env = env
        self.env.seed(seed)
        self.action_space = self.env.action_space
        self.state_space = self.env.observation_space
        self.curr_obs = self.env.reset()
        self.is_done = False
        self.visual = visual
        self.stack_size = stack_size
        
    def step(self, action):
        if not self.visual:
            if isinstance(self.action_space, spaces.Discrete):
                # We encode actions in finite spaces as an integer inside a length-1 array
                # but Gym wants the integer itself
                assert action.ndim == 1 and action.size == 1 and action.dtype in (np.int32, np.int64)
                action = action[0]
            else:
                assert action.ndim == 1 and action.dtype in (np.float32, np.float64)

            next_obs, reward, self.is_done, _ = self.env.step(action)
            return next_obs, reward
        else:
            assert action.ndim == 1 and action.dtype in (np.float32, np.float64)

            next_obs, reward, self.is_done, _ = self.env.step(action)
            return next_obs, reward

    
    def action_gen(self, policy):
        if policy:
            action = policy.get_output((self.curr_obs[np.newaxis, :])).squeeze()
            if len(action.shape)==0:
                action = np.array([action])
        else:
            action = self.env.action_space.sample()
        return action
    
    def reset(self):
        if not self.visual:
            self.curr_obs = self.env.reset()
        else:
            self.env.reset()
            ob_temp = rgb2gray(self.env.render(mode='rgb_array'))[...,None]/255.0 # Add the first rendered image
            ob = ob_temp
            for _ in range(self.stack_size-1):
                ob = np.concatenate((ob,ob_temp),axis=-1)#[x+y+z for x,y,z in zip (ob,ob,ob)])
            self.curr_obs = ob
        self.is_done = False
        
    def play_one_episode(self, policy):
        self.reset()
        #####curr_ob = self.env.render(mode='rgb_array')[...,:3]
        #####curr_obs = []
        #####curr_obs.append(curr_ob)
        transitions, actions, rewards = [], [], []
        ts = 0
        while not self.is_done and ts<1000:
            action = self.action_gen(policy)
            next_obs, reward = self.step(action)
            #####curr_ob = self.env.render(mode='rgb_array')[...,:3]
            #####curr_obs.append(curr_ob)
            if not self.visual:
                transitions.append(np.concatenate((self.curr_obs, next_obs), axis=0))
                actions.append(action)
                rewards.append(reward)
                self.curr_obs = next_obs
            else:
                transitions.append(np.concatenate((self.curr_obs, rgb2gray(self.env.render(mode='rgb_array'))[...,None]/255.0), axis=-1))
                actions.append(action)
                rewards.append(reward)
                self.curr_obs = np.append(self.curr_obs[...,1:], rgb2gray(self.env.render(mode='rgb_array'))[...,None]/255.0, axis=-1)
            #r_sum += self.step(action.squeeze())
            ts += 1
        #####np.savez('obs.npz', obs = np.array(curr_obs))
        #####stop
        return transitions, actions, rewards, ts
    
    def evaluate(self, eval_ntarjs, nn):
        rew_sum = []
        ts_list = []
        for i in range(eval_ntarjs):
            _, _, rewards, ts= self.play_one_episode(nn)
            rew_sum.append(np.sum(rewards))
            ts_list.append(ts)
        return np.mean(rew_sum), np.mean(ts_list)
    

