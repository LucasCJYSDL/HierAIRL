# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .maze_env import MazeEnv
from .ant import AntEnv
import numpy as np
import gym


class _AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv
    
    def __init__(self, env_name="AntMaze", top_down_view=False):
        manual_collision = False
        env_name = env_name[3:]
        maze_size_scaling = 8
        maze_id = None
        observe_blocks = False
        put_spin_near_agent = False
        if env_name == 'Maze':
            maze_id = 'Maze'
        elif env_name == 'Push':
            maze_id = 'Push'
        elif env_name == 'Fall':
            maze_id = 'Fall'
        elif env_name == 'Block':
            maze_id = 'Block'
            put_spin_near_agent = True
            observe_blocks = True
        elif env_name == 'BlockMaze':
            maze_id = 'BlockMaze'
            put_spin_near_agent = True
            observe_blocks = True
        else:
            raise ValueError('Unknown maze environment %s' % env_name)

        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'n_bins': 0,
            'observe_blocks': observe_blocks,
            'put_spin_near_agent': put_spin_near_agent,
            'top_down_view': top_down_view,
            'manual_collision': manual_collision,
            'maze_size_scaling': maze_size_scaling
        }
        super(_AntMazeEnv, self).__init__(**gym_mujoco_kwargs)
        self.reset()

    def get_pos(self):
        return self.wrapped_env.physics.data.qpos[:3].copy()


class AntMazeEnv_v0(_AntMazeEnv):
    def __init__(self):
        self.available_sub_goals = np.asarray(((0., 0.), (16., 0.), (16., 16.), (0., 16.)), dtype=np.float32)
        self.current_goal = 0
        super(AntMazeEnv_v0, self).__init__("AntMaze", top_down_view=True)

    def reset(self):
        super(AntMazeEnv_v0, self).reset()
        self.current_goal = np.random.randint(0, 3)
        self.wrapped_env.set_xy(self.available_sub_goals[self.current_goal] + np.random.rand(2) * 4. - 2.)
        return np.concatenate((self._get_obs(), self.available_sub_goals[-1]))

    def _forward_reward(self, last_pos, current_pos):

        if self.current_goal == -1:
            return 0., True
        reached = False
        rwd = 0.
        current_goal = self.available_sub_goals[self.current_goal+1]
        if self.current_goal in (0, 1, 2) and np.linalg.norm(current_pos[:2] - current_goal) < 4.:
            self.current_goal += 1
        if np.linalg.norm(current_pos[:2] - self.available_sub_goals[-1]) < 4.:
            self.current_goal = -1
            reached = True
            rwd += 100.
        last_d = np.linalg.norm(last_pos[:2] - current_goal)
        d = np.linalg.norm(current_pos[:2] - current_goal)
        rwd += last_d - d
        return rwd, reached

    def step(self, a):
        last_pos = self.get_pos()
        obs, _, _, info = super(AntMazeEnv_v0, self).step(a)
        current_pos = self.get_pos()
        done = (not np.isfinite(current_pos).all() or current_pos[2] <= 0.3 or current_pos[2] >= 1.1)
        forward_reward, reached = self._forward_reward(last_pos, current_pos)
        ctl_reward = -5.e-3 * np.square(a).sum()
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = ctl_reward
        return np.concatenate((obs, self.available_sub_goals[-1])), (forward_reward + ctl_reward), (done or reached), info

    @property
    def observation_space(self):
        shape = (len(self._get_obs())+2,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


class AntMazeEnv_v1(_AntMazeEnv):
    def __init__(self):
        self.available_sub_goals = [np.asarray(((0., 0.), (16., 0.), (16., 16.), (0., 16.)), dtype=np.float32),
                                    np.asarray(((0., 16.), (16., 16.), (16., 0.), (0., 0.)), dtype=np.float32)]
        self.current_goal = 0
        self.sub_task = 0
        super(AntMazeEnv_v1, self).__init__("AntMaze", top_down_view=True)

    def reset(self):
        super(AntMazeEnv_v1, self).reset()
        self.sub_task = np.random.randint(0, 2)
        self.current_goal = np.random.randint(0, 3)
        self.wrapped_env.set_xy(self.available_sub_goals[self.sub_task][self.current_goal] + np.random.rand(2) * 4. - 2.)
        return np.concatenate((self._get_obs(), self.available_sub_goals[self.sub_task][-1]))

    def _forward_reward(self, last_pos, current_pos):
        if self.current_goal == -1:
            return 0., True
        reached = False
        rwd = 0.
        current_goal = self.available_sub_goals[self.sub_task][self.current_goal+1]
        if self.current_goal in (0, 1, 2) and np.linalg.norm(current_pos[:2] - current_goal) < 4.:
            self.current_goal += 1
        if np.linalg.norm(current_pos[:2] - self.available_sub_goals[self.sub_task][-1]) < 4.:
            self.current_goal = -1
            reached = True
            rwd += 100.
        last_d = np.linalg.norm(last_pos[:2] - current_goal)
        d = np.linalg.norm(current_pos[:2] - current_goal)
        rwd += last_d - d
        return rwd, reached

    def step(self, a):
        last_pos = self.get_pos()
        obs, _, _, info = super(AntMazeEnv_v1, self).step(a)
        current_pos = self.get_pos()
        done = (not np.isfinite(current_pos).all() or current_pos[2] <= 0.3 or current_pos[2] >= 1.1)
        forward_reward, reached = self._forward_reward(last_pos, current_pos)
        ctl_reward = -5.e-3 * np.square(a).sum()
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = ctl_reward
        return np.concatenate((obs, self.available_sub_goals[self.sub_task][-1])), (forward_reward + ctl_reward), (done or reached), info

    @property
    def observation_space(self):
        shape = (len(self._get_obs())+2,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


class AntPushEnv_v0(_AntMazeEnv):
    def __init__(self):
        self.current_goal = 0
        self.available_sub_goals = np.asarray(((0., 0.), (-8., 0.), (-8., 8.), (0., 8.), (0., 19.)), dtype=np.float32)
        super(AntPushEnv_v0, self).__init__("AntPush", top_down_view=True)

    def reset(self):
        super(AntPushEnv_v0, self).reset()
        self.current_goal = np.random.randint(0, 3)
        self.wrapped_env.set_xy(self.available_sub_goals[self.current_goal] + np.random.rand(2) * 4. - 2.)
        return np.concatenate((self._get_obs(), self.available_sub_goals[-1]))

    def _forward_reward(self, last_pos, current_pos):
        if self.current_goal == -1:
            return 0., True
        reached = False
        rwd = 0.
        current_goal = self.available_sub_goals[self.current_goal+1]
        if self.current_goal in (0, 1, 2) and np.linalg.norm(current_pos[:2] - current_goal) < 3.5:
            self.current_goal += 1
        if np.linalg.norm(current_pos[:2] - self.available_sub_goals[-1]) < 4.:
            self.current_goal = -1
            reached = True
            rwd += 100.
        last_d = np.linalg.norm(last_pos[:2] - current_goal)
        d = np.linalg.norm(current_pos[:2] - current_goal)
        rwd += last_d - d
        return rwd, reached

    def step(self, a):
        last_pos = self.get_pos()
        obs, _, _, info = super(AntPushEnv_v0, self).step(a)
        current_pos = self.get_pos()
        forward_reward, reached = self._forward_reward(last_pos, current_pos)
        ctl_reward = -5.e-3 * np.square(a).sum()
        done = (reached or current_pos[2] <= 0.3 or current_pos[2] >= 1.0)
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = ctl_reward
        return np.concatenate((obs, self.available_sub_goals[-1])), (forward_reward + ctl_reward), done, info

    @property
    def observation_space(self):
        shape = (len(self._get_obs())+2,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


class AntPushEnv_v1(_AntMazeEnv):
    def __init__(self):
        self.current_goal = 0
        self.available_sub_goals = [
            np.asarray(((0., 0.), (-8., 0.), (-8., 8.), (0., 8.), (0., 19.)), dtype=np.float32),
            np.asarray(((-8., 8.), (-8., 0.), (0., 0.), (0., 8.), (11., 8.)), dtype=np.float32)]
        self.sub_task = 0
        super(AntPushEnv_v1, self).__init__("AntPush", top_down_view=True)

    def reset(self):
        super(AntPushEnv_v1, self).reset()
        self.sub_task = np.random.randint(0, 2)
        self.current_goal = np.random.randint(0, 3)
        self.wrapped_env.set_xy(self.available_sub_goals[self.sub_task][self.current_goal] + np.random.rand(2) * 4. - 2.)
        return np.concatenate((self._get_obs(), self.available_sub_goals[self.sub_task][-1]))

    def _forward_reward(self, last_pos, current_pos):
        if self.current_goal == -1:
            return 0., True
        reached = False
        rwd = 0.
        current_goal = self.available_sub_goals[self.sub_task][self.current_goal+1]
        if self.current_goal in (0, 1, 2) and np.linalg.norm(current_pos[:2] - current_goal) < 3.5:
            self.current_goal += 1
        if np.linalg.norm(current_pos[:2] - self.available_sub_goals[self.sub_task][-1]) < 4.:
            self.current_goal = -1
            reached = True
            rwd += 100.
        last_d = np.linalg.norm(last_pos[:2] - current_goal)
        d = np.linalg.norm(current_pos[:2] - current_goal)
        rwd += last_d - d
        return rwd, reached

    def step(self, a):
        last_pos = self.get_pos()
        obs, _, _, info = super(AntPushEnv_v1, self).step(a)
        current_pos = self.get_pos()
        forward_reward, reached = self._forward_reward(last_pos, current_pos)
        ctl_reward = -5.e-3 * np.square(a).sum()
        done = (reached or current_pos[2] <= 0.3 or current_pos[2] >= 1.0)
        info["reward_forward"] = forward_reward
        info["reward_ctrl"] = ctl_reward
        return np.concatenate((obs, self.available_sub_goals[self.sub_task][-1])), (forward_reward + ctl_reward), done, info

    @property
    def observation_space(self):
        shape = (len(self._get_obs())+2,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


class AntMazeEnv_v(_AntMazeEnv):
    def __init__(self):
        self.goal = np.asarray((0., 16.), dtype=np.float32)
        super(AntMazeEnv_v, self).__init__("AntMaze", top_down_view=True)

    def reset(self):
        obs = super(_AntMazeEnv, self).reset()
        return np.concatenate((obs, self.goal))

    def step(self, a):
        obs, _, _, info = super(_AntMazeEnv, self).step(a)
        reward = -np.linalg.norm(obs[:2] - self.goal)
        done = (reward > -4.)
        return np.concatenate((obs, self.goal)), reward, done, info

    @property
    def observation_space(self):
        shape = (len(self._get_obs()) + 2,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


class AntPushEnv_v(_AntMazeEnv):
    def __init__(self):
        self.goal = np.asarray((0., 19.), dtype=np.float32)
        super(AntPushEnv_v, self).__init__("AntPush", top_down_view=True)

    def reset(self):
        obs = super(_AntMazeEnv, self).reset()
        return np.concatenate((obs, self.goal))

    def step(self, a):
        obs, _, _, info = super(_AntMazeEnv, self).step(a)
        reward = -np.linalg.norm(obs[:2] - self.goal)
        done = (reward > -4.)
        return np.concatenate((obs, self.goal)), reward, done, info

    @property
    def observation_space(self):
        shape = (len(self._get_obs()) + 2,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


class AntFallEnv_v(_AntMazeEnv):
    def __init__(self):
        self.goal = np.asarray((0., 27., 4.5), dtype=np.float32)
        super(AntFallEnv_v, self).__init__("AntFall", top_down_view=False)

    def reset(self):
        obs = super(_AntMazeEnv, self).reset()
        return np.concatenate((obs, self.goal))

    def step(self, a):
        obs, _, _, info = super(_AntMazeEnv, self).step(a)
        reward = -np.linalg.norm(obs[:3] - self.goal)
        done = (reward > -4.)
        return np.concatenate((obs, self.goal)), reward, done, info

    @property
    def observation_space(self):
        shape = (len(self._get_obs()) + 3,)
        high = np.inf * np.ones(shape, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low, high)


def run_environment(env_name, episode_length, num_episodes):
    env = AntMazeEnv()

    def action_fn(obs):
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = (action_space_mean +
                         action_space_magn *
                         np.random.uniform(low=-1.0, high=1.0,
                                           size=action_space.shape))
        return random_action

    rewards = []
    successes = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        for _ in range(episode_length):
            env.render()
            obs, reward, done, _ = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            if done:
                break
        print('Episode %d reward: %.2f, Success: %d', ep + 1, rewards[-1], successes[-1])

    print('Average Reward over %d episodes: %.2f',
                 num_episodes, np.mean(rewards))
    print('Average Success over %d episodes: %.2f',
                 num_episodes, np.mean(successes))


def main():
    run_environment("AntMaze", 1000, 100)


if __name__ == '__main__':
    main()
