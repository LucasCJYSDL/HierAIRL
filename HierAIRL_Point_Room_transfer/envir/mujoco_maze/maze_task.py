"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import random

from envir.mujoco_maze.maze_env_utils import MazeCell


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 5.0,
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float: # so the first 2 dimensions have to be (x, y)
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]
    swimmer: Optional[float]


class MazeTask(ABC):
    REWARD_THRESHOLD: float
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=4.0, swimmer=4.0)
    INNER_REWARD_SCALING: float = 0.0

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale

    def sample_goals(self) -> bool:
        return False

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                print("Great Success!!!")
                return True
        return False

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class DistRewardMixIn:
    REWARD_THRESHOLD: float = -1000.0
    goals: List[MazeGoal]
    scale: float

    def reward(self, obs: np.ndarray) -> float:
        return -self.goals[0].euc_dist(obs) / self.scale


class GoalReward4Rooms(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0005
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([8.0 * scale, -8.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        # return [
        #     [B, B, B, B, B, B, B, B, B],
        #     [B, E, E, E, B, E, E, E, B],
        #     [B, E, E, E, E, E, E, E, B],
        #     [B, E, E, E, B, E, E, E, B],
        #     [B, B, E, B, B, B, E, B, B],
        #     [B, E, E, E, B, E, E, E, B],
        #     [B, E, E, E, E, E, E, E, B],
        #     [B, R, R, R, B, E, E, E, B],
        #     [B, B, B, B, B, B, B, B, B],
        # ]

        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, B, B, B, B, B, B, E, E, B, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, E, E, B, B, B, B, E, E, B, E, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, B, E, B, E, E, R, E, E, B, E, B, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, E, B, E, E, B, B, B, B, E, E, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, B, B, B, B, B, B, B, B, B, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]


class DistReward4Rooms(GoalReward4Rooms, DistRewardMixIn):
    REWARD_THRESHOLD: float = -1000.0  # ???

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([4.0 * scale, 4.0 * scale]))]
        self.mid_goals = [MazeGoal(np.array([-2.0 * scale, 4.0 * scale])),
                          MazeGoal(np.array([-4.0 * scale, 4.0 * scale])),
                          MazeGoal(np.array([-4.0 * scale, -4.0 * scale])),
                          MazeGoal(np.array([4.0 * scale, -4.0 * scale])),
                          MazeGoal(np.array([4.0 * scale, 4.0 * scale]))]
        self.goal_idx = 0

    def reward(self, pre_obs: np.ndarray, obs: np.ndarray) -> float:
        ori_rwd = super(DistReward4Rooms, self).reward(obs)
        pre_dist = self.mid_goals[self.goal_idx].euc_dist(pre_obs)
        cur_dist = self.mid_goals[self.goal_idx].euc_dist(obs)
        rwd = 0.0
        # rwd += (pre_dist - cur_dist) / self.scale
        # rwd += min(1.0 / cur_dist, 1.0)
        if cur_dist < 2.0:
            print("Arrive at Goal {}".format(self.goal_idx))
            if self.goal_idx < len(self.mid_goals) - 1:
                self.goal_idx += 1
                rwd += 20.0
        # print(rwd + ori_rwd * 1000.0)
        return ori_rwd * 100.0 + rwd

    def expert_action(self, cur_obs):
        x = cur_obs[0]
        y = cur_obs[1]
        ori = cur_obs[2]
        goal_x = self.mid_goals[self.goal_idx].pos[0]
        goal_y = self.mid_goals[self.goal_idx].pos[1]
        tang = (goal_y - y) / (goal_x - x)
        target_ori = np.arctan(tang)
        if (goal_x - x) < 0:
            if (goal_y - y) < 0:
                target_ori -= np.pi
            else:
                target_ori += np.pi
        act = np.array([1.0, target_ori - ori])
        vel_noise = np.random.rand()
        ori_noise = 0.2 * np.random.rand() - 0.1
        act[0] -= vel_noise
        act[1] += ori_noise

        if self.goal_idx == 0:
            option = 0
        else:
            option = self.goal_idx - 1

        return act, option



# mainly to test the prior
class GoalRewardCorridor(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.0005
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=4.0, swimmer=4.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        # (-1, 5), (-5, -1), (1, -5), (5, 1)
        self.goals = [MazeGoal(np.array([-3.0 * scale, 7.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, B, E, E, E, E, E, E, B],
            [B, E, B, B, E, B, E, B, B, B, B, E, B],
            [B, E, B, B, E, B, E, B, B, B, B, E, B],
            [B, E, B, B, B, B, E, B, B, E, E, R, B],
            [B, E, B, B, B, E, E, E, B, B, B, B, B],
            [B, E, E, E, E, E, R, E, E, E, E, R, B],
            [B, B, B, B, B, E, E, E, B, B, B, E, B],
            [B, E, E, E, B, B, E, B, B, B, B, E, B],
            [B, E, B, B, B, B, E, B, E, B, B, E, B],
            [B, E, B, B, B, B, E, B, E, B, B, E, B],
            [B, E, E, E, E, E, E, B, R, E, E, R, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]



class DistRewardCorridor(GoalRewardCorridor):
    REWARD_THRESHOLD: float = -1000.0 # ???
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([-7.0 * scale, -3.0 * scale]))]
        self.mid_goals = [MazeGoal(np.array([0.0 * scale, -3.0 * scale])), MazeGoal(np.array([-5.0 * scale, -3.0 * scale])), MazeGoal(np.array([-5.0 * scale, 2.0 * scale])),
                          MazeGoal(np.array([-10.0 * scale, 2.0 * scale])), MazeGoal(np.array([-10.0 * scale, -3.0 * scale])),
                          MazeGoal(np.array([-7.0 * scale, -3.0 * scale]))]
        # self.mid_goals = [MazeGoal(np.array([-3.0 * scale, 5.0 * scale]))]
        self.goal_idx = 0

    def reward(self, pre_obs: np.ndarray, obs: np.ndarray) -> float:
        ori_rwd = super(DistRewardCorridor, self).reward(obs)
        pre_dist = self.mid_goals[self.goal_idx].euc_dist(pre_obs)
        cur_dist = self.mid_goals[self.goal_idx].euc_dist(obs)
        rwd = 0.0
        # rwd += (pre_dist - cur_dist) / self.scale
        # rwd += min(1.0 / cur_dist, 1.0)
        if cur_dist < 2.0:
            print("Arrive at Goal {}".format(self.goal_idx))
            if self.goal_idx < len(self.mid_goals) - 1:
                self.goal_idx += 1
                rwd += 20.0
        # print(rwd + ori_rwd * 1000.0)
        return rwd + ori_rwd * 100.0

    def expert_action(self, cur_obs):
        x = cur_obs[0]
        y = cur_obs[1]
        ori = cur_obs[2]
        goal_x = self.mid_goals[self.goal_idx].pos[0]
        goal_y = self.mid_goals[self.goal_idx].pos[1]
        tang = (goal_y - y) / (goal_x - x)
        target_ori = np.arctan(tang)
        if (goal_x - x) < 0:
            if (goal_y - y) < 0:
                target_ori -= np.pi
            else:
                target_ori += np.pi
        act = np.array([1.0, target_ori - ori])
        vel_noise = np.random.rand()
        ori_noise = 0.2 * np.random.rand() - 0.1
        act[0] -= vel_noise
        act[1] += ori_noise

        if self.goal_idx == 0 or self.goal_idx == 4:
            option = 0
        elif self.goal_idx == 1 or self.goal_idx == 3:
            option = 1
        elif self.goal_idx == 2:
            option = 2
        else:
            option = 3

        return act, option


class NewDistRewardCorridor(GoalRewardCorridor):
    REWARD_THRESHOLD: float = -1000.0 # ???
    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([-10.0 * scale, 7.0 * scale]))]
        self.mid_goals = [MazeGoal(np.array([0.0 * scale, -3.0 * scale])), MazeGoal(np.array([-5.0 * scale, -3.0 * scale])),
                          MazeGoal(np.array([-5.0 * scale, 7.0 * scale])), MazeGoal(np.array([-10.0 * scale, 7.0 * scale]))]
        # self.mid_goals = [MazeGoal(np.array([-3.0 * scale, 5.0 * scale]))]
        self.goal_idx = 0

    def reward(self, pre_obs: np.ndarray, obs: np.ndarray) -> float:
        ori_rwd = super(NewDistRewardCorridor, self).reward(obs)
        pre_dist = self.mid_goals[self.goal_idx].euc_dist(pre_obs)
        cur_dist = self.mid_goals[self.goal_idx].euc_dist(obs)
        rwd = 0.0
        # rwd += (pre_dist - cur_dist) / self.scale
        # rwd += min(1.0 / cur_dist, 1.0)
        if cur_dist < 2.0:
            print("Arrive at Goal {}".format(self.goal_idx))
            if self.goal_idx < len(self.mid_goals) - 1:
                self.goal_idx += 1
                rwd += 20.0
        # print(rwd + ori_rwd * 1000.0)
        return rwd + ori_rwd * 100.0

    def expert_action(self, cur_obs):
        x = cur_obs[0]
        y = cur_obs[1]
        ori = cur_obs[2]
        goal_x = self.mid_goals[self.goal_idx].pos[0]
        goal_y = self.mid_goals[self.goal_idx].pos[1]
        tang = (goal_y - y) / (goal_x - x)
        target_ori = np.arctan(tang)
        if (goal_x - x) < 0:
            if (goal_y - y) < 0:
                target_ori -= np.pi
            else:
                target_ori += np.pi
        act = np.array([1.0, target_ori - ori])
        vel_noise = np.random.rand()
        ori_noise = 0.2 * np.random.rand() - 0.1
        act[0] -= vel_noise
        act[1] += ori_noise

        if self.goal_idx == 0:
            option = 0
        elif self.goal_idx == 1 or self.goal_idx == 3:
            option = 1
        elif self.goal_idx == 2:
            option = 2
        else:
            option = 3

        return act, option

class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "4Rooms": [GoalReward4Rooms, DistReward4Rooms],
        "Corridor": [GoalRewardCorridor, DistRewardCorridor, NewDistRewardCorridor]
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
