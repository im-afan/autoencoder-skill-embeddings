import project_config
import logger

import os
import copy
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gymnasium as gym
#import roboschool

from PPO import PPO
import stable_baselines3
from stable_baselines3.common.noise import NormalActionNoise

################################### Training ###################################
def train():
    env = gym.make(project_config.ENV_NAME, render_mode="rgb_array", terminate_when_unhealthy=True)
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]

    action_noise = NormalActionNoise(action_dim, sigma=0.1*np.ones(action_dim))
    model = stable_baselines3.TD3("MlpPolicy", env, action_noise=action_noise, verbose=3)
    model.learn(total_timesteps=100000, log_interval=100)
    #model.save("tq3_ant_pretrained")
    print("========= training finished ==========")

    vec_env = model.get_env()
    obs = vec_env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = vec_env.step(action)
        vec_env.render("human")
        if(done):
            print("done dddd")
            vec_env.reset()

if __name__ == '__main__':
    train()