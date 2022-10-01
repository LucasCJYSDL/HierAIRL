#!/usr/bin/env python3
import numpy as np
import torch.multiprocessing as multiprocessing
from utils.common_utils import set_seed
from utils.config import ARGConfig, Config
from utils.plot_utils import draw_traj
from default_config import mujoco_config, rlbench_config

def option_loop(env):
    a_array = []
    c_array = []
    s_array = []
    r_array = []
    st, done = env.reset(random=False), False
    while not done:
        at, ct = env.get_expert_act(st)
        s_array.append(st)
        c_array.append(ct)
        a_array.append(at)
        st, r, done = env.step(at)
        # env.render()
        r_array.append(r)

    return s_array, c_array, a_array, r_array


def plot(config: Config):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env
    elif env_type == "rlbench":
        from envir.rlbench_env import RLBenchEnv as Env
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_thread = config.n_thread
    seed = config.seed
    env_name = config.env_name

    set_seed(seed)
    env = Env(env_name)
    env.init()

    for r_id in range(20):
        states, options, _, _ = option_loop(env)
        states_array = np.array(states)
        options_array = np.array(options)
        x_array = states_array[:, 0]
        y_array = states_array[:, 1]

        traj_array = {0: [{'x': [], 'y': []}], 1: [{'x': [], 'y': []}], 2: [{'x': [], 'y': []}], 3: [{'x': [], 'y': []}], 4: [{'x': [], 'y': []}]}
        for i in range(len(options_array)):
            tmp_option = int(options_array[i])
            tmp_x = x_array[i]
            tmp_y = y_array[i]
            traj_array[tmp_option][0]['x'].append(tmp_x)
            traj_array[tmp_option][0]['y'].append(tmp_y)
        draw_traj(env_name, option_num=4, trajectory_list=traj_array, unique_token=env_name+'exp', episode_id=r_id)



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    arg.add_arg("env_name", "PointCorridor-v1", "Environment name") # only for Point4Rooms-v1 or PointCorridor-v1
    arg.add_arg("seed", 0, "Random seed")
    arg.parser()

    if arg.env_type == "rlbench":
        config = rlbench_config
    elif arg.env_type == "mujoco":
        config = mujoco_config
    else:
        raise ValueError("rlbench for rlbench env; mujoco for mujoco env")

    config.update(arg)

    plot(config)