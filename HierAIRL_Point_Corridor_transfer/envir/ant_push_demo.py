from envir import mujoco_env
import gym
import time
import numpy as np
from envir import mujoco_maze

def main():
    env = gym.make('PointCorridor-v1')
    # for _ in range(10):
    obs = env.reset()
    print("1: ", obs, obs.shape, env.observation_space.shape)
    # (30, )
    r_array = []
    for _ in range(1000):
        # action = env.action_space.sample()
        action, option = env.get_expert_action(obs)
        env.render()
        time.sleep(0.02)
        print("2: ", option, obs[:2])
        # (8, )
        obs, r, done, info = env.step(action)
        r_array.append(r)
        print("3: ", obs, r, done)
        if done:
            break
    print(np.array(r_array).sum())

if __name__ == '__main__':
    main()