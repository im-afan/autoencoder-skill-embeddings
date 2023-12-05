from typing import Optional

import numpy as np

from gymnasium.spaces import Box

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.flip import Flip
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.reach import Reach
from panda_gym.envs.tasks.slide import Slide
from panda_gym.envs.tasks.stack import Stack
from panda_gym.pybullet import PyBullet

import custom_envs.pick_and_place_3dgoal
import custom_envs.pick
from custom_envs.custom_reach import ReachWithGripper
from movement_autoencoder import Decoder
import project_config

import torch

import logger
import time

class ReachWithGripperLowLevel(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "joint",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        logging: bool = False,
        render: bool = False
    ) -> None:
        #print("aklsdjfhalksdjhfladksjfhaklsdhf")
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = ReachWithGripper(sim, reward_type=reward_type, get_ee_position=self.robot.get_ee_position, get_fingers_width=self.robot.get_fingers_width)
        self.logging = logging
        super().__init__(
            self.robot,
            self.task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
    
    def step(self, action):
        orig_state = self.robot.get_obs()
        ret = super().step(action)
        new_state = self.robot.get_obs()
        if(self.logging):
            print(orig_state, new_state, action, self.task.cur_time)
            logger.log_state(orig_state, new_state, action, self.task.cur_time)
        return ret

class ReachWithGripperHighLevel(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "joint",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        logging: bool = False,
        render: bool = False
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = ReachWithGripper(sim, reward_type=reward_type, get_ee_position=self.robot.get_ee_position, get_fingers_width=self.robot.get_fingers_width)
        self.logging = logging

        #print("action space shape: ", self.robot.action_space.shape[0])
        #print("observation space shape: ", self.robot.get_obs().shape[0])
        #print()
        #print() 

        self.decoder = Decoder(self.robot.get_obs().shape[0], self.robot.action_space.shape[0], project_config.AUTOENCODER_LATENT_SIZE_PANDA)
        decoder_path = project_config.DECODER_PATH
        with open("cur_path.txt", "r") as f:
            decoder_path = f.readline()
            decoder_path += "/autoencoders/decoder.pth"
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder = self.decoder.double()

        super().__init__(
            self.robot,
            self.task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

        self.action_space = Box(
            -np.ones((project_config.AUTOENCODER_LATENT_SIZE_PANDA)),
            np.ones((project_config.AUTOENCODER_LATENT_SIZE_PANDA))
        )
    
    def step(self, a):
        #print(a)
        #time.sleep(0.01)
        a = torch.tensor(a)
        orig_state = self.robot.get_obs()
        action = self.decoder(torch.tensor(orig_state), a).detach().numpy()
        #print(action)
        ret = super().step(action)
        new_state = self.robot.get_obs()
        if(self.logging):
            print(orig_state, new_state, action, self.task.cur_time)
            logger.log_state(orig_state, new_state, action, self.task.cur_time)
        return ret



class PickAndPlaceLowLevel(RobotTaskEnv):
    #copied from github, need to change default settings
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "joint",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

class PickAndPlaceHighLevel(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "joint",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        logging: bool = False,
        render: bool = False
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = custom_envs.pick_and_place_3dgoal.PickAndPlace3dGoalLowLevel(sim, get_ee_position=self.robot.get_ee_position, reward_type=reward_type)
        self.logging = logging

        self.decoder = Decoder(self.robot.get_obs().shape[0], self.robot.action_space.shape[0], project_config.AUTOENCODER_LATENT_SIZE_PANDA)
        decoder_path = project_config.DECODER_PATH
        with open("cur_path.txt", "r") as f:
            decoder_path = f.readline()
            decoder_path += "/autoencoders/decoder.pth"
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder = self.decoder.double()

        super().__init__(
            self.robot,
            self.task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

        self.action_space = Box(
            -np.ones((project_config.AUTOENCODER_LATENT_SIZE_PANDA)),
            np.ones((project_config.AUTOENCODER_LATENT_SIZE_PANDA))
        )
    
    def step(self, a):
        #print(a)
        #time.sleep(0.01)
        a = torch.tensor(a)
        orig_state = self.robot.get_obs()
        action = self.decoder(torch.tensor(orig_state), a).detach().numpy()
        #print(action)
        ret = super().step(action)
        new_state = self.robot.get_obs()
        #if(self.logging):
        #    print(orig_state, new_state, action, self.task.cur_time)
        #    logger.log_state(orig_state, new_state, action, self.task.cur_time)
        return ret

class PickUpLowLevel(RobotTaskEnv):
    #copied from github, need to change default settings
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "joint",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        logging: bool = False,
        render: bool = False,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = custom_envs.pick.PickUp(sim, get_ee_position=self.robot.get_ee_position, reward_type=reward_type)
        super().__init__(
            self.robot,
            self.task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


class PickUpHighLevel(RobotTaskEnv):
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "joint",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        logging: bool = False,
        render: bool = False
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = custom_envs.pick.PickUp(sim, get_ee_position=self.robot.get_ee_position, reward_type=reward_type)
        self.logging = logging

        self.decoder = Decoder(self.robot.get_obs().shape[0], self.robot.action_space.shape[0], project_config.AUTOENCODER_LATENT_SIZE_PANDA)
        decoder_path = project_config.DECODER_PATH
        with open("cur_path.txt", "r") as f:
            decoder_path = f.readline()
            decoder_path += "/autoencoders/decoder.pth"
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.decoder = self.decoder.double()

        super().__init__(
            self.robot,
            self.task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

        self.action_space = Box(
            -np.ones((project_config.AUTOENCODER_LATENT_SIZE_PANDA)),
            np.ones((project_config.AUTOENCODER_LATENT_SIZE_PANDA))
        )
    
    def step(self, a):
        #print(a)
        a = torch.tensor(a)
        orig_state = self.robot.get_obs()
        action = self.decoder(torch.tensor(orig_state), a).detach().numpy()
        #print(action)
        ret = super().step(action)
        new_state = self.robot.get_obs()
        #if(self.logging):
        #    print(orig_state, new_state, action, self.task.cur_time)
        #    logger.log_state(orig_state, new_state, action, self.task.cur_time)
        return ret


