from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance
import time

class ReachWithGripper(Task):
    #basic reach task but with retracting/extending the gripper
    def __init__(
        self,
        sim,
        get_ee_position,
        get_fingers_width,
        reward_type="dense",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.get_fingers_width = get_fingers_width
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0, 0.08])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range, 0.08])
        self.cur_time = 0
        self.finger_retract_time_low = 0
        self.finger_retract_time_high = 100 
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([self.finger_retract_time, self.cur_time]) 

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        finger_width = np.array([self.get_fingers_width()])
        return np.concatenate([ee_position, finger_width])
        #return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.cur_time = 0
        self.finger_retract_time = np.random.randint(self.finger_retract_time_low, self.finger_retract_time_high)
        #print(self.goal[:3].shape)
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        #print("hjere asdkfjasdhf")
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        time.sleep(0.1)
        self.cur_time += 1
        #print(achieved_goal, desired_goal)
        if(self.cur_time >= self.finger_retract_time):
            self.goal[-1] = 0
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)