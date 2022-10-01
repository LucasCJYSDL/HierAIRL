#!/usr/bin/env python3
import os
import copy
import torch
from typing import Union
import matplotlib.pyplot as plt
import torch.multiprocessing as multiprocessing
from model.option_ppo import OptionPPO, PPO
from model.option_gail import OptionGAIL, GAIL
from model.option_airl import OptionAIRL, AIRL
from utils.common_utils import validate, reward_validate, get_dirs, set_seed
from sampler import Sampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config, rlbench_config
from vae_pretrain import pretrain
import datetime


def make_il(config: Config, dim_s, dim_a):
    use_option = config.use_option

    if use_option:
        if config.is_airl:
            il = OptionAIRL(config, dim_s=dim_s, dim_a=dim_a)
        else:
            il = OptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = OptionPPO(config, il.policy)
    else:
        if config.is_airl:
            il = AIRL(config, dim_s=dim_s, dim_a=dim_a)
        else:
            il = GAIL(config, dim_s=dim_s, dim_a=dim_a)
        ppo = PPO(config, il.policy)
    return il, ppo


def train_g(ppo: Union[OptionPPO, PPO], sample_sxar, factor_lr):

    ppo.step(sample_sxar, lr_mult=factor_lr)


def train_d(il: Union[OptionGAIL, GAIL, OptionAIRL, AIRL], sample_sxar, demo_sxar, n_step=10):

    il.step(sample_sxar, demo_sxar, n_step=n_step)


def sample_batch(il: Union[OptionGAIL, GAIL, OptionAIRL, AIRL], agent, n_sample, demo_sa_array):
    demo_sa_in = agent.filter_demo(demo_sa_array)
    sample_sxar_in = agent.collect(il.policy.state_dict(), n_sample, fixed=False)
    sample_sxar, sample_rsum, sample_rsum_max = il.convert_sample(sample_sxar_in) # replace the real environment reward with the one generated with IL
    demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, demo_sxar, sample_rsum, sample_rsum_max, demo_rsum


def learn(config: Config, msg="default"):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo
    elif env_type == "rlbench":
        from envir.rlbench_env import RLBenchEnv as Env, get_demo
    else:
        raise ValueError(f"Unknown env type {env_type}")

    use_option = config.use_option
    n_demo = config.n_demo
    n_sample = config.n_sample
    n_thread = config.n_thread
    n_pre_epoch = config.n_pretrain_epoch
    n_epoch = config.n_epoch
    seed = config.seed
    pre_log_interval = config.pretrain_log_interval
    env_name = config.env_name
    use_d_info_gail = config.use_d_info_gail

    set_seed(seed)
    log_dir, save_dir, sample_name, pretrain_name = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config)) # important for reproducing and visualisaton
    logger = Logger(log_dir) # tensorboard
    save_name_pre_f = lambda i: os.path.join(save_dir, f"pre_{i}.torch")
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()

    demo = get_demo(config.copy(), path=sample_name, n_demo=n_demo, display=False)

    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a)

    demo_sa_array = tuple((s.to(il.device), c.to(il.device), a.to(il.device)) for s, c, a, r in demo)

    if use_d_info_gail:
        if os.path.isfile(pretrain_name):
            print(f"Loading pre-train model from {pretrain_name}")
            param = torch.load(pretrain_name)
            il.policy.load_state_dict(param)
        else:
            pretrain(il.policy, demo_sa_array, save_name_pre_f,
                     logger, msg, n_pre_epoch, pre_log_interval)

        temp_state_dict = copy.deepcopy(il.policy.option_policy.state_dict())
        config.use_vae = False
        il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a)

        il.policy.option_policy.load_state_dict(temp_state_dict) # only pretrain the high-level policy, so the low-level policy should be recovered

    sampling_agent = Sampler(seed, env, il.policy, n_thread=n_thread)
    sample_sxar, demo_sxar, sample_r, sample_r_max, demo_r = sample_batch(il, sampling_agent, n_sample, demo_sa_array)
    # print(sample_sxar[0], demo_sxar[0])

    v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
    info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)

    # TODO
    # if use_option:
    #     a = plt.figure()
    #     a.gca().plot(cs_demo[0][1:])
    #     logger.log_test_fig("expert_c", a, 0)
    #
    #     a = plt.figure()
    #     a.gca().plot(cs_sample[0][1:]) # plot the trajectory with the highest return
    #     logger.log_test_fig("sample_c", a, 0)

    logger.log_test_info(info_dict, 0)
    print(f"init: r-sample-avg={sample_r}, r-demo-avg={demo_r}, log_p={v_l} ; {msg}")

    for i in range(n_epoch):
        print("Starting collecting samples......")
        time_s = datetime.datetime.now()
        sample_sxar, demo_sxar, sample_r, sample_r_max, demo_r = sample_batch(il, sampling_agent, n_sample, demo_sa_array) # n_sample is too big
        time_e = datetime.datetime.now()
        print("Time cost: ", (time_e - time_s).seconds)
        if i % 3 == 0:
            train_d(il, sample_sxar, demo_sxar)
        # factor_lr = lr_factor_func(i, 1000., 1., 0.0001) # not commented by me
        train_g(ppo, sample_sxar, factor_lr=1.)
        if (i + 1) % config.log_interval == 0:
            v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
            logger.log_test("expert_logp", v_l, i)
            info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)
            # TODO
            # if use_option:
            #     a = plt.figure()
            #     a.gca().plot(cs_demo[0][1:])
            #     logger.log_test_fig("expert_c", a, i)
            #
            #     a = plt.figure()
            #     a.gca().plot(cs_sample[0][1:])
            #     logger.log_test_fig("sample_c", a, i)

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
        time_e = datetime.datetime.now()
        print("Total time cost for this epoch: ", (time_e - time_s).seconds)


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    arg.add_arg("env_name", "PointCorridor-v1", "Environment name")
    arg.add_arg("algo", "option_gail", "which algorithm to use, can be [gail, option_gail, DI_gail, airl, option_airl]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "Random seed")

    arg.add_arg("n_pretrain_epoch", 1000, "Pre-training epochs")
    arg.add_arg("n_demo", 1000, "Number of demonstration s-a")
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
        print(f"Training RLBench.* envs with larger policy network size :{config.hidden_policy}")

    ## set up the corresponding algorithm
    if config.algo == 'gail' or config.algo == 'airl':
        config.use_option = False
        config.use_c_in_discriminator = False
        config.use_d_info_gail = False
        config.use_vae = False
        config.train_option = False
    elif config.algo == 'option_gail' or config.algo == 'option_airl':
        config.use_option = True
        config.use_c_in_discriminator = True
        config.use_d_info_gail = False
        config.use_vae = False
        config.train_option = True
    elif config.algo == 'DI_gail':
        config.use_option = True
        config.use_c_in_discriminator = False
        config.use_d_info_gail = True
        config.use_vae = True
        config.train_option = False
    else:
        raise NotImplementedError

    if 'airl' in config.algo:
        config.is_airl = True
    else:
        config.is_airl = False

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")

    learn(config, msg=config.tag)
