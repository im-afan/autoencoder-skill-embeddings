import gymnasium as gym
import panda_gym
import numpy as np
import torch
import time
from register_envs import register_envs
register_envs()

env = gym.make("ReachWithGripperHighLevel-v0", render_mode="human")
#env = gym.make("PickUpLowLevel-v0", render_mode="human")
#env = gym.make("ReachWithGripperLowLevel-v0", render_mode="human")
observation, info = env.reset()
print(env.action_space)

for _ in range(200):
    np.random.seed(_)
    action = np.zeros_like(env.action_space.sample()) # random action
    #action = torch.tensor(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    #print(action)
    time.sleep(0.1)

    if terminated or truncated:
        observation, info = env.reset()
        print("EPISODE DONE DDD\n\n")

env.close()