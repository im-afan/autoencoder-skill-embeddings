import numpy as np

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv, ant_v4
from gymnasium.spaces import Box

class HighLevelEnvWrapper(gym.Env):
    """
    this ant environment takes in an embedding and takes an action by mapping this embedding to an action using the decoder
    """ 
    def __init__(self, env, decoder, latent_size, *args, **kwargs):
        self.env = env
        self.cur_frames = 0
        self.decoder = decoder
        self.latent_size = latent_size
        super().__init__(*args, **kwargs)

    def step(self, action, **kwargs):
        mapped_action = self.decoder(action)
        return self.env.step(mapped_action, **kwargs)

    def reset(self, *args, **kwargs): 
        return self.env.reset(*args, **kwargs)
