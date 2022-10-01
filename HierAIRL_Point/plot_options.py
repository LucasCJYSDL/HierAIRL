#!/usr/bin/env python3
import os
import torch
from typing import Union
import matplotlib.pyplot as plt
import torch.multiprocessing as multiprocessing
from model.MHA_option_ppo import MHAOptionPPO
from model.MHA_option_il import MHAOptionAIRL, MHAOptionGAIL
from utils.common_utils import validate, reward_validate, get_dirs, set_seed
from sampler import Sampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config, rlbench_config
from utils.plot_utils import draw_traj


def make_il(config: Config, dim_s, dim_a):
    if config.is_airl:
        il = MHAOptionAIRL(config, dim_s=dim_s, dim_a=dim_a)
    else:
        il = MHAOptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
    ppo = MHAOptionPPO(config, il.policy)

    return il, ppo

def sample_batch(il: Union[MHAOptionGAIL, MHAOptionAIRL], agent, n_sample, demo_sa_array):
    demo_sa_in = agent.filter_demo(demo_sa_array)
    sample_sxar_in = agent.collect(il.policy.state_dict(), n_sample, fixed=False)
    sample_sxar, sample_rsum = il.convert_sample(sample_sxar_in) # replace the real environment reward with the one generated with IL
    demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, demo_sxar, sample_rsum, demo_rsum

def plot(config: Config, msg="default", policy_dir=''):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo
    elif env_type == "rlbench":
        from envir.rlbench_env import RLBenchEnv as Env, get_demo
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_thread = config.n_thread
    seed = config.seed
    env_name = config.env_name

    set_seed(seed)
    log_dir, save_dir, sample_name, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    # with open(os.path.join(save_dir, "config.log"), 'w') as f:
    #     f.write(str(config))  # important for reproducing and visualisaton

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()

    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a)

    il.load_state_dict(torch.load(policy_dir, map_location=config.device))

    sampling_agent = Sampler(seed, env, il.policy, n_thread=n_thread)
    for r_id in range(20):
        info_dict, cs_sample, trajs = reward_validate(sampling_agent, il.policy, do_print=True, n_sample=-1, return_traj=True)
        states, options, _, _ = trajs[0]
        states_array = states.detach().cpu().numpy()
        options_array = options.detach().cpu().numpy()
        x_array = states_array[:, 0]
        y_array = states_array[:, 1]

        traj_array = {0: [{'x': [], 'y': []}], 1: [{'x': [], 'y': []}], 2: [{'x': [], 'y': []}], 3: [{'x': [], 'y': []}], 4: [{'x': [], 'y': []}]}
        for i in range(len(options_array)-1):
            tmp_option = int(options_array[i+1][0])
            tmp_x = x_array[i]
            tmp_y = y_array[i]

            traj_array[tmp_option][0]['x'].append(tmp_x)
            traj_array[tmp_option][0]['y'].append(tmp_y)
        draw_traj(env_name, option_num=4, trajectory_list=traj_array, unique_token=env_name, episode_id=r_id)



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    arg.add_arg("env_name", "PointCorridor-v1", "Environment name") # only for Point4Rooms-v1 or PointCorridor-v1
    arg.add_arg("algo", "hier_airl", "which algorithm to use, can be [option_airl, hier_airl, hier_gail]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "Random seed")
    arg.add_arg("n_demo", 5000, "Number of demonstration s-a")
    arg.parser()

    if arg.env_type == "rlbench":
        config = rlbench_config
    elif arg.env_type == "mujoco":
        config = mujoco_config
    else:
        raise ValueError("rlbench for rlbench env; mujoco for mujoco env")

    config.update(arg)
    if config.env_name.startswith("Humanoid"):
        config.hidden_policy = (512, 512)
        config.hidden_critic = (512, 512)
        print(f"Training Humanoid.* envs with larger policy network size :{config.hidden_policy}")
    if config.env_type == "rlbench":
        config.hidden_policy = (128, 128)
        config.hidden_option = (128, 128)
        config.hidden_critic = (128, 128)
        config.log_clamp_policy = (-20., -2.)
        config.dmodel = 80
        config.mha_nhid = 100
        print(f"Training RLBench.* envs with larger policy network size :{config.hidden_policy}")

    if 'airl' in config.algo:
        config.is_airl = True
    else:
        config.is_airl = False

    if 'hier' in config.algo:
        config.use_posterior = True
    else:
        config.use_posterior = False

    config.use_c_in_discriminator = True
    config.use_vae = False

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")

    if 'Corridor' in config.env_name:
        ckpt_pth = './plot_ckpt/corridor/1954.torch'
    else:
        ckpt_pth = './plot_ckpt/corridor/1799.torch'

    plot(config, msg=config.tag, policy_dir=ckpt_pth)