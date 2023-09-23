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

from movement_autoencoder import Autoencoder

class MovementDataset(Dataset):
    def __init__(self, data, transform=None, checkpoint_path="Autoencoder_pretrained/"):
        self.data = data
        self.transfom = transform
        self.checkpoint_path = checkpoint_path

    def __getitem__(self, index):
        orig_state = self.data[index].orig_state
        end_state = self.data[index].end_state
        action = self.data[index].action
        return orig_state, end_state, action

    def __len__(self):
        return len(self.data)

################################### Training ###################################
def train(env):
    epochs = 10
    #env_name = project_config.ENV_NAME
    has_continuous_action_space = True;
    #env = gym.make(env_name)
    checkpoint_path = "./autoencoder_pretrained/"

    print_freq = 10                # print frequenecy (loss, etc)
    save_model_freq = 1000          # save model frequency (in num timesteps)
    start_time = datetime.now()


    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    autoencoder = Autoencoder(state_dim, action_dim, 8)
    dataset = MovementDataset(logger.logged_states)
    loader = DataLoader(dataset, batch_size=16)

    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    running_loss = 0

    for epoch in range(epochs):
        for batch_index, (begin_state, end_state, action) in enumerate(loader):
            #print("index: {}, orig: {}, end: {}".format(batch_index, begin_state.shape, end_state.shape))
            optimizer.zero_grad()

            output = autoencoder(begin_state, action)

            loss = loss_fn(output, action)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            if(batch_index % print_freq == print_freq-1):
                print("epoch : {}, batch index : {}, loss : {}".format(epoch, batch_index, running_loss))
                running_loss = 0
        # save model
        print("--------------------------SAVING MODEL-------------------------")
        print("saving encoder model at: " + checkpoint_path + "encoder.pth")
        print("saving decoder model at: " + checkpoint_path + "decoder.pth")
        autoencoder.save(checkpoint_path + "encoder.pth")
        print("save successful")
        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print("---------------------------------------------------------------")

if __name__ == '__main__':
    train()
