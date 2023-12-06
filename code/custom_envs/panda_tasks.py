from typing import Optional

import numpy as np

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
from custom_envs.custom_reach import ReachWithGripper

import logger

from movement_autoencoder import Decoder

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
        #print("aklsdjfhalksdjhfladksjfhaklsdhf")
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = ReachWithGripper(sim, reward_type=reward_type, get_ee_position=self.robot.get_ee_position, get_fingers_width=self.robot.get_fingers_width)
        self.logging = logging
        self.decoder = Decoder()
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
        reward_type: str = "sparse",
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
        #print("render mode asdfkadf", render_mode)
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = custom_envs.pick_and_place_3dgoal.PickAndPlaceHighLevel(sim, reward_type=reward_type)
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