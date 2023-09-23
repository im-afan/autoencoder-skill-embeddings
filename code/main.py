import torch
import numpy as np
import train_lowlevel, train_autoencoder
import movement_autoencoder
import gymnasium as gym
import project_config
from agent import Agent
from custom_envs.ant_turn import CustomAntEnv

def main():
    ####### initialize environment hyperparameters ######
    env_name = project_config.ENV_NAME 
    sample_timesteps_per_agent = 10000
    agent_train_timesteps = 1000
    agent_log_timesteps = 10
    agent_save_timestes = 10

    has_continuous_action_space = True  # continuous action space; else discrete
 
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    env = CustomAntEnv()
    agent_lowlevel = Agent(env)
    """agent_lowlevel.train(
        agent_train_timesteps,
        agent_log_timesteps,
        agent_save_timestes
    )"""
    agent_lowlevel.sample_movement(sample_timesteps_per_agent)
    train_lowlevel.sample_data()
    train_autoencoder.train()

if(__name__ == "__main__"):
    main()

