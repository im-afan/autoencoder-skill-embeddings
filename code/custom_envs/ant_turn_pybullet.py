from math import sqrt
import pybullet
import numpy as np
import pybullet_envs_gymnasium
from pybullet_envs_gymnasium.env_bases import MJCFBaseBulletEnv
from pybullet_envs_gymnasium.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs_gymnasium.robot_locomotors import Ant, WalkerBase
from pybullet_envs_gymnasium.gym_locomotion_envs import WalkerBaseBulletEnv
from custom_envs.custom_walker import WalkerTargetPosBulletEnv
import project_config
from logger import log_state
import time
import torch
from torch import nn
from gymnasium.spaces import Box
from movement_autoencoder import Decoder
from pybullet_envs_gymnasium.scene_stadium import StadiumScene

class AntTargetPosBulletEnv(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = Ant()
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)

class AntTurnHighLevelEnv(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = Ant()
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)
        
        try:
            state_dict = torch.load(kwargs["decoder_path"])
        except:
            state_dict = torch.load("./autoencoder_pretrained_size2/decoder.pth")
        print(WalkerTargetPosBulletEnv)
        self.decoder = Decoder(self.observation_space.shape[0],
                               self.action_space.shape[0], 
                               project_config.AUTOENCODER_LATENT_SIZE)
        self.decoder.load_state_dict(state_dict)

        self.action_space = Box(
            np.zeros(project_config.AUTOENCODER_LATENT_SIZE), 
            np.ones(project_config.AUTOENCODER_LATENT_SIZE)
        )

        
    def step(self, a):
        state = torch.tensor(self.robot.calc_state())
        latent = torch.tensor(a)
        action = self.decoder(state, latent).detach().numpy()
        #print(action)
        return super().step(action)


if __name__ == "__main__":
    print(AntTargetPosBulletEnv().reset())
