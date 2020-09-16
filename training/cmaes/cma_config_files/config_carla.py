"""Test config file for cma-es code."""

executable = '/home/boschaustin/projects/CL_AD/ES/carla_lbc_new/training/train_image_ES.py'

exec_kwargs = {'--model_path': '/home/boschaustin/projects/CL_AD/ES/carla_lbc_new/training/ckpts/image/model-10.th'}

exec_args = []

pre_value_args = []

wait_limit = 3600

log_enabled = False

GPU_list = [1]

jobs_per_GPU = 1

env_seed = 0
# using 0 seed, the most common one used by LbC; same seed creates less variation in evaluation; note either CARLA or
# the LbC model have their own stochasticity, independent of this seed
# TODO use the same seed for the policy if it's probabilistic
