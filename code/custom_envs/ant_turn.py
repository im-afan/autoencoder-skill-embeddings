import numpy as np

from gym import utils
from gym.envs.mujoco import MuJocoPyEnv, ant_v4
from gym.spaces import Box

class CustomAntEnv(ant_v4):
    def __init__(self):
        super.__init__(self)

    def reset(self):
        super.reset(self)