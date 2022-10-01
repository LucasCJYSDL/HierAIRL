from envir import mujoco_env
import gym
import time
from envir import mujoco_maze

def main():
    env = gym.make('Point4Rooms-v1')
    for _ in range(10):
        obs = env.reset(is_train=True)
    print("1: ", obs, obs.shape, env.observation_space.shape)
    # (30, )
    for _ in range(1000):
        # action = env.action_space.sample()
        action, option = env.get_expert_action(obs)
        env.render()
        time.sleep(0.02)
        print("2: ", option, obs[:2])
        # (8, )
        obs, r, done, info = env.step(action)
        print("3: ", obs, r, done)
        if done:
            break

if __name__ == '__main__':
    main()