import project_config
import logger

import os
import copy
import glob
import time
from datetime import datetime

import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

import gym
#import roboschool

from PPO import PPO
from movement_autoencoder import Autoencoder

class MovementDataset(Dataset):
    def __init__(self, data, transform=None, checkpoint_path="Autoencoder_pretrained/"):
        self.data = data
        self.transfom = transform
        self.checkpoint_path = checkpoint_path

    def __getitem__(self, index):
        x = self.data[index].orig_state
        y = self.data[index].end_state
        return x, y

    def __len__(self):
        return len(self.data)

################################### Training ###################################
def train():
    ###################################  ###################################
    epochs = 10
    env_name = project_config.ENV_NAME
    has_continuous_action_space = True;
    env = gym.make(env_name)
    checkpoint_path = "autoencoder_pretrained/"

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    autoencoder = Autoencoder(state_dim, 32)
    dataset = MovementDataset(logger.logged_states)
    loader = DataLoader(dataset, batch_size=16)

    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0

    for epoch in range(epochs):
        for batch_index, (begin_state, end_state) in enumerate(loader):
            optimizer.zero_grad()

            output = autoencoder(begin_state, end_state)

            loss = loss_fn(output, end_state)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            if(batch_index % 1000 == 999):
                print("epoch: {}, batch index: {}, loss: {}", epochs, batch_index, running_loss)
                autoencoder.save(checkpoint_path=checkpoint_path)
                running_loss = 0

if __name__ == '__main__':
    train()
