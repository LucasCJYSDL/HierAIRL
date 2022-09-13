from .ant_maze_env import AntPushEnv_v0
from .ant_maze_env import AntPushEnv_v1
from .ant_maze_env import AntMazeEnv_v0
from .ant_maze_env import AntMazeEnv_v1
import gym

gym.envs.register(id="AntPush-v0",
                  entry_point="envir.ant_maze_env_ex.ant_maze_env:AntPushEnv_v0",
                  max_episode_steps=1000)


