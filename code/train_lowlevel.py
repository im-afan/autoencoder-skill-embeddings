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

import stable_baselines3
from stable_baselines3.common.noise import NormalActionNoise

from custom_envs.ant_turn import CustomAntEnv

from agent import Agent

################################### Training ###################################
def train(agent):
    sample_render_timesteps = 1000
    sample_timesteps_per_agent = 100000
    agent_train_timesteps = int(1e6)
    agent_log_timesteps = 100
    agent_save_timesteps = 1

    agent_lowlevel = Agent(env, save_path="./sb3_pretrained/low_level1")
    agent_lowlevel.train(
        agent_train_timesteps,
        agent_log_timesteps,
        agent_save_timesteps
    )

def sample_data(agent):
    agent.sample_movement(int(1000), render=True)

if __name__ == '__main__':
    env = CustomAntEnv(render_mode="rgb_array")
    agent = Agent(env, save_path="./sb3_pretrained/lowlevel1")
    train(agent)