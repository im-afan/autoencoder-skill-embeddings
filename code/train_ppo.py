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

from agent import Agent

################################### Training ###################################
def train():
    env = gym.make(project_config.ENV_NAME, render_mode="rgb_array", terminate_when_unhealthy=True)
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    
    agent = Agent(env)
    agent.policy.load("./sb3_pretrained.zip")
    agent.train(1000, 10, 10)
    agent.sample_movement(int(1000), render=True)

if __name__ == '__main__':
    train()