import os
import random

import torch
try:
    import pybullet_envs
except ImportError:
    print("Warning: pybullet not installed, bullet environments will be unavailable")
import gym
from envir import mujoco_maze


class MujocoEnv(object):
    def __init__(self, task_name: str = "HalfCheetah-v2"):
        self.task_name = task_name
        self.env = None
        self.display = False

    def init(self, display=False):
        self.env = gym.make(self.task_name)
        self.display = display
        return self

    def reset(self, random: bool = False):
        s = self.env.reset()
        return s

    def get_expert_act(self, obs):
        act, option = self.env.get_expert_action(obs)
        return act, option

    def render(self):
        self.env.render()

    def step(self, a):
        s, reward, terminate, info = self.env.step(a)
        if self.display:
            self.env.render()
        return s, reward, terminate

    def state_action_size(self):
        if self.env is not None:
            s_dim = self.env.observation_space.shape[0]
            a_dim = self.env.action_space.shape[0]
        else:
            env = gym.make(self.task_name)
            s_dim = env.observation_space.shape[0]
            a_dim = env.action_space.shape[0]
            env.close()
        return s_dim, a_dim


def get_demo(config, path="", n_demo=2048, display=False):
    assert os.path.isfile(path)
    print(f"Demo Loaded from {path}")
    samples = torch.load(path) # TODO
    random.shuffle(samples)
    n_current_demo = 0
    sample = []
    for traj in samples:
        s, c, a, r = traj
        print(s.shape, c.shape, a.shape, r.shape)
        sample.append(traj)
        n_current_demo += traj[2].size(0)
        if n_current_demo >= n_demo:
            break
    if n_current_demo < n_demo:
        print(f"Warning, demo package contains less demo than required ({n_current_demo}/{n_demo})")
    return sample

def collect_demo(config, n_demo=2048, display=False, is_manual=False, env_name=None, expert_path=None):
    from model.option_policy import Policy, OptionPolicy

    # you must have an expert model first, by running 'option_ppo_learn.py'.

    if not is_manual:
        env = MujocoEnv(config.env_name)
        path = f"./{config.env_name}_sample.torch"
    else:
        env = MujocoEnv(env_name)
        path = f"./{env_name}_sample.torch"
    dim_s, dim_a = env.state_action_size()
    env.init(display=display)

    if not is_manual:
        config.device = 'cpu'
        policy_state = torch.load(expert_path, map_location='cuda:0')
        policy = Policy(config, dim_s, dim_a)
        # policy = OptionPolicy(config, dim_s, dim_a)
        policy.load_state_dict(policy_state)

    sample = []
    n_current_demo = 0
    while n_current_demo < n_demo:
        with torch.no_grad():
            s_array = []
            a_array = []
            c_array = []
            r_array = []
            s, done = env.reset(), False
            ct = torch.tensor(4, dtype=torch.long).unsqueeze(dim=0)
            c_array.append(ct.clone())
            while not done:
                st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(dim=0)
                s_array.append(st.clone())
                if not is_manual:
                    at = policy.sample_action(st, fixed=True) # eliminate the randomness of the expert policy
                else:
                    at, ct = env.get_expert_act(obs=st.clone().numpy()[0])
                    at = torch.tensor(at, dtype=torch.float32, device=st.device).unsqueeze(dim=0)
                    ct = torch.tensor(ct, dtype=torch.long, device=st.device).unsqueeze(dim=0)
                a_array.append(at.clone())
                c_array.append(ct.clone())
                s, r, done = env.step(at.squeeze(dim=0).cpu().detach().clone().numpy())
                r_array.append(r)
            a_array = torch.cat(a_array, dim=0)
            c_array = torch.cat(c_array, dim=0).unsqueeze(dim=1)
            s_array = torch.cat(s_array, dim=0)
            r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=1)

            print(f"R-Sum={r_array.sum()}, L={r_array.size(0)}")
            # keep = input(f"{n_current_demo}/{n_demo} Keep this ? [y|n]>>>")
            # if keep == 'y':
            if r_array.sum().item() > 1000:
                sample.append((s_array, c_array, a_array, r_array))
                n_current_demo += r_array.size(0)
    torch.save(sample, path)
    return sample


def get_demo_stat(path=""):
    if os.path.isfile(path):
        print(f"Demo Loaded from {path}")
        samples, _ = torch.load(path) # TODO
        random.shuffle(samples)
        n_current_demo = 0
        sample = []
        aver_r = 0
        for traj in samples:
            s, a, r = traj
            print(s.shape, a.shape, r.shape, r.sum())
            aver_r += r.sum()
            sample.append(traj)
            n_current_demo += traj[2].size(0)
        print(aver_r/len(sample), n_current_demo, len(sample))
        return sample

if __name__ == '__main__':

    # get_demo_stat(path="/home/wenqi/Proj_2_HAIRL/Hier AIRL/data/mujoco/Hopper-v2_sample.torch")
    # get_demo_stat(path="./Walker2d-v2_sample.torch")
    # get_demo_stat(path="/home/wenqi/Proj_2_HAIRL/Hier AIRL/data/rlbench/CloseMicrowave2_sample.torch")
    # collect_demo(config=None, n_demo=10000, is_manual=True, env_name='Point4Rooms-v1')
    # get_demo_stat(path="/home/wenqi/Proj_2_HAIRL/Hier AIRL/data/mujoco/Point4Rooms-v1_sample.torch")
    collect_demo(config=None, n_demo=10000, is_manual=True, env_name='PointCorridor-v1')
    # get_demo_stat(path="/home/wenqi/Proj_2_HAIRL/Hier AIRL/data/mujoco/Ant-v2_sample.torch")
    # get_demo_stat(path="./Hopper-v2_sample.torch")
    # get_demo_stat(path="./Striker-v2_sample.torch")
    # get_demo_stat(path="/home/wenqi/Proj_2_HAIRL/Hier AIRL/data/mujoco/AntPush-v0_sample.torch")

    # import torch.multiprocessing as multiprocessing
    # from utils.config import Config, ARGConfig
    # from default_config import mujoco_config, rlbench_config
    #
    # multiprocessing.set_start_method('spawn')
    #
    # arg = ARGConfig()
    # arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    # arg.add_arg("env_name", "Striker-v2", "Environment name")
    # arg.add_arg("algo", "ppo", "Environment type, can be [ppo, option_ppo]")
    # arg.add_arg("device", "cuda:0", "Computing device")
    # arg.add_arg("tag", "default", "Experiment tag")
    # arg.add_arg("seed", 0, "Random seed")
    # arg.parser()
    #
    # config = mujoco_config if arg.env_type == "mujoco" else rlbench_config
    # config.update(arg)
    # # config.n_sample = 1024
    # if config.env_name.startswith("Humanoid"):
    #     config.hidden_policy = (512, 512)
    #     config.hidden_critic = (512, 512)
    #     print(f"Training Humanoid.* envs with larger policy network size :{config.hidden_policy}")
    # if config.env_type == "rlbench":
    #     config.hidden_policy = (128, 128)
    #     config.hidden_option = (128, 128)
    #     config.hidden_critic = (128, 128)
    #     config.log_clamp_policy = (-20., -2.)
    #     print(f"Training RLBench.* envs with larger policy network size :{config.hidden_policy}")
    #
    # print(config.algo)
    # config.use_option = True
    # config.use_c_in_discriminator = False  # in fact, there are no discriminators
    # config.use_d_info_gail = False
    # config.use_vae = False
    # config.train_option = True
    # if config.algo == 'ppo':
    #     config.use_option = False
    #     config.train_option = False
    #
    # collect_demo(config, n_demo=10000, expert_path='/home/wenqi/Proj_2_HAIRL/Hier AIRL/envir/model/striker/8349.torch')