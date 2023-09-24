import torch
import numpy as np

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv, ant_v4
from gymnasium.spaces import Box
from copy import deepcopy

class HighLevelAntEnv(ant_v4.AntEnv):
    """
    this ant environment takes in an embedding and takes an action by mapping this embedding to an action using the decoder
    """ 
    def __init__(self, decoder, latent_size, *args, **kwargs):
        self.cur_frames = 0
        self.decoder = decoder
        self.latent_size = latent_size
        self.action_space = gym.spaces.Box(np.zeros((latent_size)), np.ones((latent_size)))
        self.cur_obs = None
        super().__init__(*args, **kwargs)

    def step(self, action, **kwargs):
        mapped_action = self.decoder(
            torch.tensor(self.cur_obs, dtype=torch.float32), 
            torch.tensor(action, dtype=torch.float32)
        )
        mapped_action = mapped_action.detach().numpy()
        #print(len(super().step(mapped_action, **kwargs)))
        obs, reward, done, info, _ = super().step(mapped_action, **kwargs)
        self.cur_frames += 1
        self.cur_obs = deepcopy(obs)
        return obs, reward, done or self.cur_frames > 1000, info, _

    def reset(self, *args, **kwargs): 
        self.cur_frames = 0
        res = super().reset(*args, **kwargs)
        obs, _ = res
        self.cur_obs = deepcopy(obs)
        return super().reset(*args, **kwargs)
