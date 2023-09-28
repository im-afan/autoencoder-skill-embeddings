import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv, ant_v4
from gymnasium.spaces import Box



class CustomAntEnv(ant_v4.AntEnv):
    def __init__(self, *args, **kwargs):
        self.cur_frames = 0
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        obs, reward, done, _, info = super().step(*args, **kwargs)
        self.cur_frames += 1
        done = done or self.cur_frames > 1000
        return obs, reward, done, _, info
    def reset(self, *args, **kwargs): 
        self.cur_frames = 0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + 0.1 * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        #print(type(observation))
        return observation, None

if(__name__ == "__main__"):
    env = CustomAntEnv()
    env.reset()