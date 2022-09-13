from envir import mujoco_env
import gym
import time

def main():
    env = gym.make('AntPush-v0')
    obs = env.reset()
    print("1: ", obs, obs.shape, env.observation_space.shape)
    # (30, )
    for _ in range(1000):
        action = env.action_space.sample()
        env.render()
        time.sleep(0.02)
        print("2: ", action, action.shape)
        # (8, )
        obs, r, done, info = env.step(action)
        print("3: ", obs, r, done)
        if done:
            break

if __name__ == '__main__':
    main()