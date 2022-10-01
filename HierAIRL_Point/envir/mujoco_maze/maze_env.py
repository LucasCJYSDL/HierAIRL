"""
Mujoco Maze environment.
Based on `models`_ and `rllab`_.

.. _models: https://github.com/tensorflow/models/tree/master/research/efficient-hrl
.. _rllab: https://github.com/rll/rllab
"""

import itertools as it
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, List, Optional, Tuple, Type

import gym
import random
import numpy as np

from envir.mujoco_maze import maze_env_utils, maze_task
from envir.mujoco_maze.agent_model import AgentModel

# Directory that contains mujoco xml files.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"


class MazeEnv(gym.Env):
    def __init__(
        self,
        model_cls: Type[AgentModel],
        maze_task: Type[maze_task.MazeTask] = maze_task.MazeTask,
        include_position: bool = True,
        maze_height: float = 0.5,
        maze_size_scaling: float = 4.0,
        inner_reward_scaling: float = 1.0,
        restitution_coef: float = 0.8,
        task_kwargs: dict = {},
        image_shape: Tuple[int, int] = (600, 480),
        camera_move_x: Optional[float] = 0.0, # based on the location of the origin -- first robot
        camera_move_y: Optional[float] = 0.0,
        camera_zoom: Optional[float] = -0.2,
        **kwargs,
    ) -> None:
        self.t = 0  # time steps
        self._task = maze_task(maze_size_scaling, **task_kwargs)
        self._maze_height = height = maze_height
        self._maze_size_scaling = size_scaling = maze_size_scaling
        self._inner_reward_scaling = inner_reward_scaling
        self._restitution_coef = restitution_coef

        self._maze_structure = structure = self._task.create_maze()

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x # by default, the first 'R' is set as the origin
        self._init_torso_y = torso_y
        self._init_positions = [(x - torso_x, y - torso_y) for x, y in self._find_all_robots()]
        self.empty_blocks = self._collect_empty_blocks()
        # print("1: ", self.empty_blocks)
        self.sample_inits = False
        # print("init positions: ", self._init_positions)

        if model_cls.MANUAL_COLLISION:
            if model_cls.RADIUS is None:
                raise ValueError("Manual collision needs radius of the model")
            self._collision = maze_env_utils.CollisionDetector(
                structure,
                size_scaling,
                torso_x,
                torso_y,
                model_cls.RADIUS,
            )
        else:
            self._collision = None

        # Let's create MuJoCo XML
        xml_path = os.path.join(MODEL_DIR, model_cls.FILE) # based on the agent.xml
        print("XML_path: ", xml_path)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        height_offset = 0.0

        # setup the blocks
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                struct = structure[i][j]
                x, y = j * size_scaling - torso_x, i * size_scaling - torso_y
                h = height / 2 * size_scaling
                size = size_scaling * 0.5

                if struct.is_block():
                    # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {h + height_offset}",
                        size=f"{size} {size} {h}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 1",
                    )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if "name" not in geom.attrib:
                raise Exception("Every geom of the torso must have a name")

        # Set goals
        # assert len(self._task.goals) == 1, 'Currently, we only test on single-goal tasks!'
        for i, goal in enumerate(self._task.goals):
            z = goal.pos[2] if goal.dim >= 3 else 0.0
            if goal.custom_size is None:
                size = f"{maze_size_scaling * 0.1}"
            else:
                size = f"{goal.custom_size}"
            ET.SubElement(
                worldbody,
                "site",
                name=f"goal_site{i}",
                pos=f"{goal.pos[0]} {goal.pos[1]} {z}",
                size=size,
                rgba=goal.rgb.rgba_str(),
            )

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)

        self.world_tree = tree
        self.wrapped_env = model_cls(file_path=file_path, **kwargs) # tell the configuration to gym.mujoco_env
        self.observation_space = self._get_obs_space()
        self._camera_move_x = camera_move_x
        self._camera_move_y = camera_move_y
        self._camera_zoom = camera_zoom

        self._image_shape = image_shape
        self._mj_offscreen_viewer = None

    def get_expert_action(self, obs):
        return self._task.expert_action(obs)


    def set_sample_inits(self, sample_inits: bool) -> None:
        self.sample_inits = sample_inits

    def get_ori(self) -> float:
        return self.wrapped_env.get_ori()

    def get_xy(self) -> np.ndarray:
        return self.wrapped_env.get_xy()

    def _get_obs_space(self) -> gym.spaces.Box:
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        # Set velocity limits, does not consider the additional obs
        wrapped_obs_space = self.wrapped_env.observation_space
        high[: wrapped_obs_space.shape[0]] = wrapped_obs_space.high
        low[: wrapped_obs_space.shape[0]] = wrapped_obs_space.low
        # Set coordinate limits
        low[0], high[0], low[1], high[1] = self._xy_limits()
        # Set orientation limits
        return gym.spaces.Box(low, high)

    def _xy_limits(self) -> Tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = 100, 100, -100, -100
        structure = self._maze_structure
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_block():
                continue
            xmin, xmax = min(xmin, j), max(xmax, j)
            ymin, ymax = min(ymin, i), max(ymax, i)
        x0, y0 = self._init_torso_x, self._init_torso_y
        scaling = self._maze_size_scaling
        xmin, xmax = (xmin - 0.5) * scaling - x0, (xmax + 0.5) * scaling - x0
        ymin, ymax = (ymin - 0.5) * scaling - y0, (ymax + 0.5) * scaling - y0
        return xmin, xmax, ymin, ymax

    # TODO: integrate with sensor info
    def _get_obs(self) -> np.ndarray:
        wrapped_obs = self.wrapped_env._get_obs()
        # currently we don't have top_view and additional observations
        view, additional_obs = [], []

        obs = np.concatenate([wrapped_obs[:3]] + additional_obs + [wrapped_obs[3:]])
        return np.concatenate([obs, *view, np.array([self.t * 0.001])])

    def reset(self) -> np.ndarray:
        self.t = 0
        self.wrapped_env.reset()
        self._task.goal_idx = 0
        self.rwd = 0.0
        # print("1: ", self._get_obs())
        # Samples a new start position
        if self.sample_inits:
            xy = random.choice(self._init_positions)
            print("init_xy: ", xy)
            self.wrapped_env.set_xy(xy)
        return self._get_obs()

    def seed(self, seed_idx: int=None):
        super().seed(seed_idx)
        self.action_space.np_random.seed(seed_idx)
        self.wrapped_env.seed(seed_idx)
        self.wrapped_env.action_space.np_random.seed(seed_idx)
        random.seed(seed_idx)
        np.random.seed(seed_idx)
        
    def _maybe_move_camera(self, viewer: Any) -> None:
        from mujoco_py import const

        if self._camera_move_x is not None:
            viewer.move_camera(const.MOUSE_MOVE_V, self._camera_move_x, 0.0)
        if self._camera_move_y is not None:
            viewer.move_camera(const.MOUSE_MOVE_H, 0.0, self._camera_move_y)
        if self._camera_zoom is not None:
            viewer.move_camera(const.MOUSE_ZOOM, 0.0, self._camera_zoom)

    def render(self, mode="human", **kwargs) -> Optional[np.ndarray]:

        if self.wrapped_env.viewer is None:
            self.wrapped_env.render(mode, **kwargs)
            self._maybe_move_camera(self.wrapped_env.viewer)
        return self.wrapped_env.render(mode, **kwargs)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def _find_robot(self) -> Tuple[float, float]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def _find_all_robots(self) -> List[Tuple[float, float]]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        coords = []
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_robot():
                coords.append((j * size_scaling, i * size_scaling))
        return coords

    def _collect_empty_blocks(self) -> List[np.ndarray]:
        structure = self._maze_structure
        size_scaling = self._maze_size_scaling
        coords = []
        for i, j in it.product(range(len(structure)), range(len(structure[0]))):
            if structure[i][j].is_empty():
                coords.append(np.array([j * size_scaling - self._init_torso_x, i * size_scaling - self._init_torso_y]))
        return coords

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        cur_obs = self._get_obs()
        self.t += 1
        if self.wrapped_env.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()

            # Checks that the new_position is in the wall
            collision = self._collision.detect(old_pos, new_pos)
            if collision is not None:
                pos = collision.point + self._restitution_coef * collision.rest()
                if self._collision.detect(old_pos, pos) is not None:
                    # If pos is also not in the wall, we give up computing the position
                    self.wrapped_env.set_xy(old_pos)
                else:
                    self.wrapped_env.set_xy(pos)
        else:
            inner_next_obs, inner_reward, _, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        inner_reward = self._inner_reward_scaling * inner_reward  # rwd from the agent
        outer_reward = self._task.reward(cur_obs, next_obs)  # rwd from the outer task
        done = self._task.termination(next_obs)

        height = next_obs[2]
        # if height < 0.3 or height > 1.0:
        #     done = True

        info["position"] = self.wrapped_env.get_xy()
        return next_obs, outer_reward, done, info

    def set_init_state(self, state: np.ndarray):
        assert state[-1] == 0, "The final state is the initial timestamp!"
        self.wrapped_env.set_full_state(state[:-1]) # currently, there is no additional obs, so the first n-1 dims are obs from the agent
        return self._get_obs()

    def set_init_xy(self, xy: np.ndarray):
        self.wrapped_env.set_xy(xy)
        return self._get_obs()

    def close(self) -> None:
        self.wrapped_env.close()

