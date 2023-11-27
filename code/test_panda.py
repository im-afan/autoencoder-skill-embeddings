import gymnasium as gym
import panda_gym
import numpy as np
from register_envs import register_envs
register_envs()

env = gym.make("ReachWithGripperLowLevel-v0", render_mode="human")
observation, info = env.reset()

for _ in range(100000):
    current_position = observation["observation"][0:3]
    desired_position = observation["desired_goal"][0:3]
    #action = 5.0 * (desired_position - current_position)
    action = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.task.get_fingers_width())

    if terminated or truncated:
        observation, info = env.reset()

env.close()