from typing import Any, Dict

import torch
from movement_autoencoder import Decoder
from gymnasium.spaces import Box
import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from panda_gym.envs.tasks.pick_and_place import PickAndPlace

import project_config
import time

class PickUp(Task):
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.get_ee_position = get_ee_position
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.prev_fingers_width = 0
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        #print("reset asdkfjhaslkdfjhalksdfjhasdf")
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        #print(object_position)
        rot = np.array([0.0, 0.0, np.random.uniform(0, np.pi)])
        #print(rot)
        self.sim.set_base_pose("object", object_position, rot)

    def _sample_goal(self) -> np.ndarray:
        return np.array([0, 0, self.object_size/2])
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        #time.sleep(0.01)
        #print(achieved_goal)
        grip_bonus = 0
        if(achieved_goal[2] > self.object_size/2+0.01): #uhh idk if this is actually the y pos 
            grip_bonus = 10
        if self.reward_type == "sparse":
            return grip_bonus 
        else: 
            return grip_bonus - distance(achieved_goal, self.get_ee_position())
