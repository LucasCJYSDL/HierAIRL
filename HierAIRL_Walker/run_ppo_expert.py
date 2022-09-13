#!/usr/bin/env python3

import os
import torch
from typing import Union
import torch.multiprocessing as multiprocessing
from model.option_ppo import PPO, OptionPPO
from model.option_policy import OptionPolicy, Policy
from sampler import Sampler
from utils.common_utils import lr_factor_func, get_dirs, reward_validate, set_seed
from utils.logger import Logger
import matplotlib.pyplot as plt
from utils.config import Config, ARGConfig
from default_config import mujoco_config, rlbench_config


def sample_batch(policy: Union[OptionPolicy, Policy], agent, n_step):
    sample = agent.collect(policy.state_dict(), n_step, fixed=False)
    rsum = sum([sxar[-1].sum().item() for sxar in sample]) / len(sample)
    return sample, rsum


def learn(config: Config, msg="default"):
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env
    else:
        from envir.rlbench_env import RLBenchEnv as Env

    use_option = config.use_option
    env_name = config.env_name
    n_sample = config.n_sample
    n_thread = config.n_thread
    n_epoch = config.n_epoch
    seed = config.seed
    set_seed(seed)

    log_dir, save_dir, sample_name, pretrain_name = get_dirs(seed, "ppo", env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)

    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()
    print(dim_s, dim_a)

    if use_option:
        policy = OptionPolicy(config, dim_s=dim_s, dim_a=dim_a)
        ppo = OptionPPO(config, policy)
    else:
        policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
        ppo = PPO(config, policy)

    sampling_agent = Sampler(seed, env, policy, n_thread=n_thread)

    for i in range(n_epoch):
        sample_sxar, sample_r = sample_batch(policy, sampling_agent, n_sample)
        lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
        ppo.step(sample_sxar, lr_mult=lr_mult)
        if (i + 1) % 50 == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, policy) # testing performance
            # TODO
            # if cs_sample is not None:
            #     a = plt.figure()
            #     a.gca().plot(cs_sample[0][1:])
            #     logger.log_test_fig("sample_c", a, i)
            torch.save(policy.state_dict(), save_name_f(i))
            logger.log_test_info(info_dict, i)
        print(f"{i}: r-sample-avg={sample_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i) # a very important metric
        logger.flush()


if __name__ == "__main__":
    # learn the expert policy/option-policy based on the environment rewards using PPO
    # can it be used for rlbench?

    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    arg.add_arg("env_name", "Striker-v2", "Environment name")
    arg.add_arg("algo", "ppo", "Environment type, can be [ppo, option_ppo]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_epoch", 30000, "Number of training epochs")
    arg.add_arg("seed", 0, "Random seed")
    arg.parser()

    config = mujoco_config if arg.env_type == "mujoco" else rlbench_config
    config.update(arg)
    # config.n_sample = 1024
    if config.env_name.startswith("Humanoid"):
        config.hidden_policy = (512, 512)
        config.hidden_critic = (512, 512)
        print(f"Training Humanoid.* envs with larger policy network size :{config.hidden_policy}")
    if config.env_type == "rlbench":
        config.hidden_policy = (128, 128)
        config.hidden_option = (128, 128)
        config.hidden_critic = (128, 128)
        config.log_clamp_policy = (-20., -2.)
        print(f"Training RLBench.* envs with larger policy network size :{config.hidden_policy}")

    config.use_option = True
    config.use_c_in_discriminator = False # in fact, there are no discriminators
    config.use_d_info_gail = False
    config.use_vae = False
    config.train_option = True
    if config.algo == 'ppo':
        config.use_option = False
        config.train_option = False

    print(f">>>> Training {config.algo} on {config.env_name} environment, on {config.device}")
    learn(config, msg=config.tag)
