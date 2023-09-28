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
from stable_baselines3 import TD3

from custom_envs.ant_turn1 import CustomAntEnv
from custom_envs.ant_turn import AntTargetPosEnv 

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
    env = gym.make("Ant-v4", render_mode="rgb_array", use_contact_forces=True)
    #env = AntTargetPosEnv(render_mode="rgb_array", use_contact_forces=True)
    agent = Agent(env, save_path="./sb3_pretrained/lowlevel1")
    train(agent)
    #sample_data(agent)

    #env = gym.make("Ant-v4", render_mode="rgb_array")
    """env = CustomAntEnv(render_mode="rgb_array", terminate_when_unhealthy=True, use_contact_forces=True)
    agent1 = TD3.load("./sb3_pretrained/Ant-v3.zip", env=env)
    print(agent1)
    vec_env = agent1.get_env() 
    print(vec_env)
    obs = vec_env.reset()
    cur_timesteps = 0
    total_timesteps = 1000
    while cur_timesteps < total_timesteps:
        action, _ = agent1.predict(obs)
        #print(type(vec_env.step(action)))
        obs, _, done, _ = vec_env.step(action)

        vec_env.render("human")
        if(done):
            print("FINISHED AAAAAASD F")
            vec_env.reset()

        cur_timesteps += 1
        #time.sleep(0.5)"""