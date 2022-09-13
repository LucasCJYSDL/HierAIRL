import os
import numpy as np
from inspect import getmembers, isclass
from rlbench import tasks
from rlbench.environment import SUPPORTED_ROBOTS
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.environment import Environment as RLEnvironment
from rlbench.task_environment import _DT, Quaternion
from pyrep.backend.utils import suppress_std_out_and_err
import random as rnd
import torch


def get_named_class(class_name: str, model):
    all_class_dict = {}
    for o in getmembers(model):
        if isclass(o[1]):
            all_class_dict[o[0]] = o[1]

    if class_name not in all_class_dict:
        raise NotImplementedError(f"No class {class_name} found in {model.__name__} !")
    return all_class_dict[class_name]


class RLBenchEnv(object):
    ROBOT_NAME = SUPPORTED_ROBOTS.keys()
    OBSERVATION_MODE = ("state",)
    ACTION_MODE = {"joint velocity": ArmActionMode.ABS_JOINT_VELOCITY,
                   "delta joint velocity": ArmActionMode.DELTA_JOINT_VELOCITY,
                   "joint position": ArmActionMode.ABS_JOINT_POSITION,
                   "delta joint position": ArmActionMode.DELTA_JOINT_POSITION,
                   "effector position": ArmActionMode.ABS_EE_POSE_WORLD_FRAME,
                   "delta effector position": ArmActionMode.DELTA_EE_POSE_WORLD_FRAME}

    def __init__(self, task_name: str = "PlaceHangerOnRack", observation_mode: str = "state",
                 action_mode: str = "delta joint position", robot_name: str = "panda"):
        self._task_name = task_name
        self._observation_mode = observation_mode
        self._action_mode = action_mode
        self._robot_name = robot_name

        self._observation_config = ObservationConfig(left_shoulder_camera=CameraConfig(image_size=(256, 256)),
                                                     right_shoulder_camera=CameraConfig(image_size=(256, 256)),
                                                     wrist_camera=CameraConfig(image_size=(256, 256)))
        self._observation_config.set_all_low_dim(True)
        self._observation_config.set_all_high_dim(False)

        self._action_config = ActionMode(RLBenchEnv.ACTION_MODE[self._action_mode])

        self.max_step = 256
        self.env = None
        self.task = None

    def init(self, display=False):
        with suppress_std_out_and_err():
            self.env = RLEnvironment(action_mode=self._action_config, obs_config=self._observation_config,
                                     headless=not display, robot_configuration=self._robot_name, static_positions=False)
            self.env.launch()
            self.task = self.env.get_task(get_named_class(self._task_name, tasks))
        return self

    def __del__(self):
        del self.task
        if self.env is not None:
            self.env.shutdown()
        del self.env

    # def gen_demo(self, random: bool = True, ):
    #     self.task._static_positions = False
    #     seed = rnd.randint(0, 4096) if random else 0
    #     seed = 1335 # can work
    #     # seed = 0 # can not work
    #     # so the performance of the gen_demo or manuel_demo is indeed influenced by the random seed,
    #     # since the initial state may change significantly in different episodes
    #     print("seed: ", seed)
    #     np.random.seed(seed)
    #     pack = self.task.get_demos(1, True)[0] # this function seems magical
    #     s_array = []
    #     a_array = []
    #     r_array = []
    #     np.random.seed(seed)
    #     desc, obs = self.task.reset()
    #     print("1: ", obs, obs.get_low_dim_data(), len(obs.get_low_dim_data()))
    #     for o_tar in pack[1:]:
    #         print("here")
    #         action = []
    #         print(self._action_config.arm)
    #         if self._action_config.arm == ArmActionMode.ABS_JOINT_VELOCITY:
    #             action.extend((o_tar.joint_positions - obs.joint_positions) / _DT)
    #         elif self._action_config.arm == ArmActionMode.ABS_JOINT_POSITION:
    #             action.extend(o_tar.joint_positions)
    #         elif self._action_config.arm == ArmActionMode.ABS_JOINT_TORQUE:
    #             action.extend(o_tar.joint_forces)
    #             raise TypeError("Warning, abs_joint_torque is not currently supported")
    #         elif self._action_config.arm == ArmActionMode.ABS_EE_POSE_WORLD_FRAME:
    #             action.extend(o_tar.gripper_pose)
    #         elif self._action_config.arm == ArmActionMode.DELTA_JOINT_VELOCITY:
    #             v_tar = (o_tar.joint_positions - obs.joint_positions) / _DT
    #             action.extend(v_tar - obs.joint_velocities)
    #             raise TypeError("Warning, delta_joint_velocity is not currently supported")
    #         elif self._action_config.arm == ArmActionMode.DELTA_JOINT_POSITION:
    #             action.extend(o_tar.joint_positions - obs.joint_positions)
    #         elif self._action_config.arm == ArmActionMode.DELTA_JOINT_TORQUE:
    #             action.extend(o_tar.joint_forces - obs.joint_forces)
    #             raise TypeError("Warning, delta_joint_torque is not currently supported")
    #         elif self._action_config.arm == ArmActionMode.DELTA_EE_POSE_WORLD_FRAME:
    #             action.extend(o_tar.gripper_pose[:3] - obs.gripper_pose[:3])
    #             q = Quaternion(o_tar.gripper_pose[3:7]) * Quaternion(obs.gripper_pose[3:7]).conjugate
    #             action.extend(list(q))
    #
    #         # print("before: ", action)
    #         action.append(0.6 if o_tar.gripper_open > 0.9 else 0.4)
    #         # print("after: ", action)
    #         action = np.asarray(action, dtype=np.float32)
    #         obs_old = obs
    #         obs, reward, done = self.task.step(action)
    #         # print("3: ", action, obs, reward)
    #         s_array.append(obs_old.get_low_dim_data())
    #         a_array.append(action)
    #         r_array.append(reward)
    #     s_array = torch.as_tensor(s_array, dtype=torch.float32)
    #     a_array = torch.as_tensor(a_array, dtype=torch.float32)
    #     r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=-1)
    #     print("2: ", s_array.shape, a_array.shape, r_array.shape)
    #     # (104, 101), (104, 8), (104, 1)
    #     return s_array, a_array, r_array
    #
    def reset(self, random: bool = False):
        # unreasonable codes
        self.task._static_positions = not random
        if not random:
            # seed = np.random.choice([3066, 360, 1957, 2919, 468])
            seed = 360
            np.random.seed(seed)
        descriptions, obs = self.task.reset()
        self._i_step = 0
        return obs.get_low_dim_data() # TODO: check the shape of the outputs from rlbench

    def step(self, a):
        a = a.copy()
        a[-1] += 0.5 # how could this be reasonable?
        obs, reward, terminate = self.task.step(a)
        self._i_step += 1
        return obs.get_low_dim_data(), reward, self._i_step >= self.max_step or terminate # it's a good sign that they provide the rewards

    def state_action_size(self):
        if self.task is not None:
            dim_a = self.env.action_size
            dim_s = len(self.task.reset()[1].get_low_dim_data())
        else:
            with suppress_std_out_and_err():
                env = RLEnvironment(action_mode=self._action_config, obs_config=self._observation_config,
                                    headless=True, robot_configuration=self._robot_name)
                dim_a = env.action_size
                env.launch()
                task = env.get_task(get_named_class(self._task_name, tasks))
                dim_s = len(task.reset()[1].get_low_dim_data())
                del task
                env.shutdown()
                del env
        return dim_s, dim_a


def get_demo(config, path="", n_demo=2048, display=False):
    if os.path.isfile(path):
        print(f"Demo Loaded from {path}")
        sample = torch.load(path) # TODO
        sample = sample[:min(len(sample), n_demo)]
    else:
        raise FileNotFoundError(f"Demo file {path} does not exist, use ./manual_rlbench_demo.py to generate one")
    return sample


if __name__ == "__main__":
    env = RLBenchEnv(task_name="CloseMicrowave2").init(display=True)
    # env.gen_demo()
    print(env.state_action_size())
    print("done")
