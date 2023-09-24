import torch
import numpy as np
import train_lowlevel, train_autoencoder
import movement_autoencoder
import gymnasium as gym
import project_config
from agent import Agent
from custom_envs.ant_turn import CustomAntEnv
from custom_envs.ant_highlevel import HighLevelAntEnv 

def main():
    ####### initialize environment hyperparameters ######
    env_name = project_config.ENV_NAME 
    sample_render_timesteps = 1000
    sample_timesteps_per_agent = 1000
    agent_train_timesteps = 1000
    agent_log_timesteps = 1
    agent_save_timestes = 1

    has_continuous_action_space = True  # continuous action space; else discrete
 
    #env = gym.make(env_name, render_mode="rgb_array", reset_noise_scale=0.1)
    env = CustomAntEnv(render_mode="rgb_array", reset_noise_scale=0.1, terminate_when_unhealthy=True)

    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    agent_lowlevel = Agent(env)
    """agent_lowlevel.train(
        agent_train_timesteps,
        agent_log_timesteps,
        agent_save_timestes
    )"""
    #agent_lowlevel.sample_movement(sample_render_timesteps, render=True)
    agent_lowlevel.sample_movement(sample_timesteps_per_agent, render=False)
    print("==== finished sampling data ====")
    autoencoder_trainer = train_autoencoder.AutoencoderWrapper(env)
    autoencoder_trainer.train()
    autoencoder = autoencoder_trainer.autoencoder.decoder

    high_level_env = HighLevelAntEnv(autoencoder, project_config.AUTOENCODER_LATENT_SIZE, render_mode="rgb_array")
    agent_highlevel = Agent(high_level_env)
    agent_highlevel.sample_movement(sample_timesteps_per_agent, render=True)

if(__name__ == "__main__"):
    main()

