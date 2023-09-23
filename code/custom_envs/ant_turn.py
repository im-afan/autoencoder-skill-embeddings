import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv, ant_v4
from gymnasium.spaces import Box

class CustomAntEnv(ant_v4.AntEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

if(__name__ == "__main__"):
    env = CustomAntEnv()
    env.reset()