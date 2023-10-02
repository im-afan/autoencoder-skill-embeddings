from pybullet_envs_gymnasium.robot_locomotors import WalkerBase
from ant_turn_pybullet import WalkerTargetPosBulletEnv
import torch
from movement_autoencoder import Decoder
import project_config
import numpy as np
from gymnasium.spaces import Box

class AntObstacle(WalkerBase):
    foot_list = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]

    def __init__(self):
        WalkerBase.__init__(self, "./assets/ant_obstaclec.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

class AntObstacleLowLevelenv(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = AntObstacle()
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)


class AntObstacleHighLevelEnv(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = AntObstacle()
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)
        
        try:
            state_dict = torch.load(kwargs["decoder_path"])
        except:
            state_dict = torch.load("./autoencoder_pretrained/decoder.pth")
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



