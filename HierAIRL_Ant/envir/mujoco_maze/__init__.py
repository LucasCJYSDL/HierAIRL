"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""

import gym

from .ant import AntEnv
from .point import PointEnv

from .maze_task import TaskRegistry

for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        point_scale = task_cls.MAZE_SIZE_SCALING.point
        if point_scale is not None:
            # Point
            gym.envs.register(
                id=f"Point{maze_id}-v{i}",
                entry_point="envir.mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=PointEnv,
                    maze_task=task_cls,
                    maze_size_scaling=point_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

        ant_scale = task_cls.MAZE_SIZE_SCALING.ant
        if ant_scale is not None:
            # Ant
            gym.envs.register(
                id=f"Ant{maze_id}-v{i}",
                entry_point="envir.mujoco_maze.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=AntEnv,
                    maze_task=task_cls,
                    maze_size_scaling=ant_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )


__version__ = "0.2.0"
