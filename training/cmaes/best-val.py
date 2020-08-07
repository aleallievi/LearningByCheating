import numpy as np
import os

path = '/projects/agents6/faraz/rl_ifo_mujoco/Walker2d-v2_pretrain_True_pid_cmaes_rew_1.0_abs_0.0_action_0.0_seed_set_0/results/'
a = []
for f in os.listdir(path):
    if f.split('_')[0] == 'value':
        with open(path+f) as ff:
            a.append(float(ff.readline()))

print(a)
print(len(a), np.max(a))

