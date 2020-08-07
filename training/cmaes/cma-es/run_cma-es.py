import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type = str, required = True)
#parser.add_argument('--seed_path', type = str, required = True)
parser.add_argument('--seed_set', type = int, required = True)
parser.add_argument('--config_path', type = str, required = True)
#parser.add_argument('--tr_iter', type = int, required = True)
#parser.add_argument('--pop_size', type = int, required = True)
parser.add_argument('--reward_weights', nargs='+', type=float, required = True)
parser.add_argument('--dist_state_weights', nargs='+', type=float, required = True)
#parser.add_argument('--dist_type', type=str, required = True)
parser.add_argument('--dist_action_weights', nargs='+', type=float, required = True)
parser.add_argument('--run_local', default = True, type=str2bool, required = True)
parser.add_argument('--nn', default = True, type=str2bool, required = True)
parser.add_argument('--pretrain', default = True, type=str2bool)
parser.add_argument('--prop_constant', default = 0, type=float)
args = parser.parse_args()

print ('args {}'.format(args))

reward_weights = args.reward_weights

dist_state_weights = args.dist_state_weights

dist_action_weights = args.dist_action_weights

#error_type = args.dist_type
# expert = '{}_normalized'.format(args.expert)

"""
for r_w in reward_weights:
    if args.run_local:
        exec_str = 'cma-es.py --seed_file {0} --run_local /projects/agents6/brahma/exps_only_err//{1}_pretrain_{2}_pid_cmaes_rew_{3}_abs_{4}_action_{5}_seed_set_{6}_pop_size_{7}/ {8} {7} {9} --reward_weight {3} --dist_state_weight {4} --dist_action_weight {5} --env_id {1}'\
            .format(args.seed_path,
                    args.env_id,
                    args.pretrain,
                    r_w,
                    args.prop_constant * (1 - r_w),
                    0,
                    args.seed_set,
                    args.pop_size,
                    args.tr_iter,
                    args.config_path)
    else:
        exec_str = 'cma-es.py --seed_file {0} /projects/agents6/brahma/exps_only_err//{1}_pretrain_{2}_pid_cmaes_rew_{3}_abs_{4}_action_{5}_seed_set_{6}_pop_size_{7}/ {8} {7} {9} --reward_weight {3} --dist_state_weight {4} --dist_action_weight {5} --env_id {1}'\
            .format(args.seed_path,
                    args.env_id,
                    args.pretrain,
                    r_w,
                    args.prop_constant * (1 - r_w),
                    0,
                    args.seed_set,
                    args.pop_size,
                    args.tr_iter,
                    args.config_path)

    print (exec_str)
    os.system(exec_str)

"""
for r_w in reward_weights:
    for d_w in dist_state_weights:
        for t_w in dist_action_weights:
            if (r_w == 0 and d_w == 0 and t_w == 0):
                continue
            if args.nn:
                if args.run_local:
                    exec_str = 'python3 cma-es.py --nn --run_local /projects/agents6/faraz/rl_ifo_mujoco/{0}_pretrain_{1}_pid_cmaes_rew_{2}_abs_{3}_action_{4}_seed_set_{5}/ {6} --reward_weight {2} --dist_state_weight {3} --dist_action_weight {4} --env_id {0}'\
                        .format(args.env_id,
                                args.pretrain,
                                r_w,
                                d_w,
                                t_w,
                                args.seed_set,
                                args.config_path)
                else:
                    exec_str = 'python3 cma-es.py --nn /projects/agents6/faraz/rl_ifo_mujoco/{0}_pretrain_{1}_pid_cmaes_rew_{2}_abs_{3}_action_{4}_seed_set_{5}/ {6} --reward_weight {2} --dist_state_weight {3} --dist_action_weight {4} --env_id {0}'\
                        .format(args.env_id,
                                args.pretrain,
                                r_w,
                                d_w,
                                t_w,
                                args.seed_set,
                                args.config_path)
            else:
                if args.run_local:
                    exec_str = 'python3 cma-es.py --run_local /projects/agents6/faraz/rl_ifo_mujoco/{0}_pretrain_{1}_pid_cmaes_rew_{2}_abs_{3}_action_{4}_seed_set_{5}/ {6} --reward_weight {2} --dist_state_weight {3} --dist_action_weight {4} --env_id {0}'\
                        .format(args.env_id,
                                args.pretrain,
                                r_w,
                                d_w,
                                t_w,
                                args.seed_set,
                                args.config_path)
                else:
                    exec_str = 'python3 cma-es.py /projects/agents6/faraz/rl_ifo_mujoco/{0}_pretrain_{1}_pid_cmaes_rew_{2}_abs_{3}_action_{4}_seed_set_{6}/ {6} --reward_weight {2} --dist_state_weight {3} --dist_action_weight {4} --env_id {0}'\
                        .format(args.env_id,
                                args.pretrain,
                                r_w,
                                d_w,
                                t_w,
                                args.seed_set,
                                args.config_path)
     
            print (exec_str)
            os.system(exec_str)

