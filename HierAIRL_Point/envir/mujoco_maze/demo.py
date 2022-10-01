import gym
import time
import random
import numpy as np
from envir import mujoco_maze

if __name__ == '__main__':
    env_id_1 = 'Point4Rooms-v0'
    env_id_2 = 'Point4Rooms-v1'
    env_id_3 = 'PointCorridor-v1'
    env_id_4 = 'Ant4Rooms-v0'
    env_id_5 = 'Ant4Rooms-v1'
    env_id_6 = 'AntCorridor-v0'
    env_id_66 = 'AntCorridor-v1'
    env_id_7 = 'Swimmer4Rooms-v0'
    env_id_8 = 'Swimmer4Rooms-v1'
    env_id_9 = 'SwimmerCorridor-v0'
    env_id_10 = 'AntControl-v0'
    env_id_11 = 'HalfCheetahControl-v0'
    env_id_12 = 'HumanoidControl-v0'

    time_steps = 1000
    pause_time = 0.01

    env = gym.make(env_id_3)
    random.seed(1)
    np.random.seed(1)
    env.seed(1)

    print("1: ", env.action_space, env.action_space.shape, env.observation_space.shape, env.action_space.shape[0], env.observation_space.shape[0])
    print("2: ", min(env.action_space.high))

    # env.set_sample_inits(True)

    obs = env.reset()
    print("5: ", obs)


    for i in range(time_steps):
        # action = np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0])
        # env.reset()
        # action = env.action_space.sample()
        action = env.get_expert_action(obs)
        obs, reward, done, _ = env.step(action)

        print("4: ", action, obs, reward, done)
        env.render()
        time.sleep(pause_time)

        if done:
            print("success at step {}".format(i))
            break