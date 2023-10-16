from custom_envs.custom_walker import WalkerTargetPosBulletEnv
from pybullet_envs_gymnasium.robot_locomotors import WalkerBase, Humanoid
import torch
from movement_autoencoder import Decoder
import project_config
import numpy as np
from gymnasium.spaces import Box

class HumanoidTargetPosLowLevel(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = Humanoid()
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)

class HumanoidTargetPosHighLevel(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = Humanoid()
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)
        
        try:
            state_dict = torch.load(kwargs["decoder_path"])
        except:
            state_dict = torch.load("./autoencoder_pretrained/humanoid/decoder.pth")
        print(WalkerTargetPosBulletEnv)
        self.decoder = Decoder(self.observation_space.shape[0],
                               self.action_space.shape[0], 
                               project_config.AUTOENCODER_LATENT_SIZE_HUMANOID)
        self.decoder.load_state_dict(state_dict)

        self.action_space = Box(
            np.zeros(project_config.AUTOENCODER_LATENT_SIZE_HUMANOID), 
            np.ones(project_config.AUTOENCODER_LATENT_SIZE_HUMANOID)
        )

        
    def step(self, a):
        state = torch.tensor(self.robot.calc_state())
        latent = torch.tensor(a)
        action = self.decoder(state, latent).detach().numpy()
        #print(action)
        return super().step(action)

