import torch
import numpy as np
import train_ppo, test_ppo, train_autoencoder
import movement_autoencoder
import gym
import project_config

def main():
    ####### initialize environment hyperparameters ######
    env_name = project_config.ENV_NAME 

    has_continuous_action_space = True  # continuous action space; else discrete
 
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    train_ppo.train()
    train_autoencoder.train()

if(__name__ == "__main__"):
    main()

