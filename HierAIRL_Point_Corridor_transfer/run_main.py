#!/usr/bin/env python3
import os
import torch
from typing import Union
import copy
import torch.multiprocessing as multiprocessing
from model.MHA_option_ppo import MHAOptionPPO
from model.MHA_option_il import MHAOptionAIRL, MHAOptionGAIL
from utils.common_utils import validate, reward_validate, get_dirs, set_seed
from sampler import Sampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config, rlbench_config


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
    sample_sxar, sample_rsum, sample_rsum_max = il.convert_sample(sample_sxar_in) # replace the real environment reward with the one generated with IL
    demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, demo_sxar, sample_rsum, sample_rsum_max, demo_rsum

def train_g(ppo: MHAOptionPPO, sample_sxar, factor_lr, train_low=False):

    ppo.step(sample_sxar, lr_mult=factor_lr, train_low=train_low)


def train_d(il: Union[MHAOptionGAIL, MHAOptionAIRL], sample_sxar, demo_sxar, n_step=10):

    il.step(sample_sxar, demo_sxar, n_step=n_step)


def learn(config: Config, msg="default"):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo
    elif env_type == "rlbench":
        from envir.rlbench_env import RLBenchEnv as Env, get_demo
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_demo = config.n_demo
    n_sample = config.n_sample
    n_thread = config.n_thread
    n_epoch = config.n_epoch
    seed = config.seed
    env_name = config.env_name

    if 'Corridor' in env_name:
        pre_ckpt_dir = './plot_ckpt/corridor/1964.torch'
    elif 'Room' in env_name:
        pre_ckpt_dir = './plot_ckpt/room/1799.torch'
    else:
        raise NotImplementedError

    set_seed(seed)
    log_dir, save_dir, sample_name, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))  # important for reproducing and visualisaton
    logger = Logger(log_dir)  # tensorboard
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()

    demo = get_demo(config.copy(), path=sample_name, n_demo=n_demo, display=False)
    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a)
    if config.init == 1:
        print("here")
        il.load_state_dict(torch.load(pre_ckpt_dir, map_location=il.device))

    demo_sa_array = tuple((s.to(il.device), c.to(il.device), a.to(il.device)) for s, c, a, r in demo)

    sampling_agent = Sampler(seed, env, il.policy, n_thread=n_thread)
    sample_sxar, demo_sxar, sample_r, sample_r_max, demo_r = sample_batch(il, sampling_agent, n_sample, demo_sa_array)

    v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
    info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)
    logger.log_test_info(info_dict, 0)
    print(f"init: r-sample-avg={sample_r}, r-demo-avg={demo_r}, log_p={v_l} ; {msg}")

    for i in range(n_epoch):
        sample_sxar, demo_sxar, sample_r, sample_r_max, demo_r = sample_batch(il, sampling_agent, n_sample, demo_sa_array) # n_sample is too big
        if i % 3 == 0:
            train_d(il, sample_sxar, demo_sxar)
        # factor_lr = lr_factor_func(i, 1000., 1., 0.0001) # not commented by me
        if (i+1) % 1 == 0:
            train_low = True
        else:
            train_low = False
        train_g(ppo, sample_sxar, factor_lr=1., train_low=train_low)

        if (i + 1) % config.log_interval == 0:
            v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
            logger.log_test("expert_logp", v_l, i)
            info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)
            if (i + 1) % (100) == 0:
                torch.save(il.state_dict(), save_name_f(i))
            logger.log_test_info(info_dict, i)
            print(f"{i}: r-sample-avg={sample_r}, r-sample-max={sample_r_max}, r-demo-avg={demo_r}, log_p={v_l} ; {msg}")
        else:
            print(f"{i}: r-sample-avg={sample_r}, r-sample-max={sample_r_max}, r-demo-avg={demo_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i)
        logger.log_train("r-sample-max", sample_r_max, i)
        logger.log_train("r-demo-avg", demo_r, i)
        logger.flush()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    arg.add_arg("env_name", "PointCorridor-v1", "Environment name")
    arg.add_arg("algo", "hier_airl", "which algorithm to use, can be [option_airl, hier_airl, hier_gail]") # only for hier_airl in the program
    arg.add_arg("init", 1, "whether to use the pre_trained ckpt")
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

    learn(config, msg=config.tag)