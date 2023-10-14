import sys

import project_config
import logger
from custom_envs.ant_turn_pybullet import AntTargetPosLowLevel
from custom_envs.humanoid_turn import HumanoidTargetPosLowLevel

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

import gymnasium as gym
import pandas as pd

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

 
class AutoencoderWrapper:
    def __init__(
            self,
            env,
            checkpoint_path="./autoencoder_pretrained/",
            logged_states_path="./logged_states/",
            print_freq=10,
        ):
        self.checkpoint_path = checkpoint_path
        self.print_freq = print_freq
        if(len(logger.logged_states)):
            self.dataset = MovementDataset(logger.logged_states)
        else:
            logged_orig_states = np.load(logged_states_path+"logged_orig_states.npy")
            logged_end_states = np.load(logged_states_path+"logged_end_states.npy")
            logged_actions = np.load(logged_states_path+"logged_actions.npy")
            for i in range(len(logged_orig_states)):
                logger.log_state(logged_orig_states[i], logged_end_states[i], logged_actions[i], 0)
            self.dataset = MovementDataset(logger.logged_states)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.autoencoder = Autoencoder(state_dim, action_dim, project_config.AUTOENCODER_LATENT_SIZE_ANT)
    
    def train(self, epochs=20):
        optimizer = Adam(self.autoencoder.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        running_loss = 0
        start_time = datetime.now()
        loader = DataLoader(self.dataset, batch_size=16)

        for epoch in range(epochs):
            for batch_index, (begin_state, end_state, action) in enumerate(loader):
                #print("index: {}, orig: {}, end: {}".format(batch_index, begin_state.shape, end_state.shape))
                optimizer.zero_grad()

                output = self.autoencoder(begin_state, action)

                loss = loss_fn(output, action)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
                if(batch_index % self.print_freq == self.print_freq-1):
                    print("epoch : {}, batch index : {}, loss : {}".format(epoch, batch_index, running_loss))
                    running_loss = 0
            # save model
            print("--------------------------SAVING MODEL-------------------------")
            print("saving encoder model at: " + self.checkpoint_path + "encoder.pth")
            print("saving decoder model at: " + self.checkpoint_path + "decoder.pth")
            self.autoencoder.save(self.checkpoint_path)
            print("save successful")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("---------------------------------------------------------------")

    def test(self, timesteps=1000):
        pass

if __name__ == '__main__':
    try:
        checkpoint_path = sys.argv[1]
    except:
        checkpoint_path = "./autoencoder_pretrained/ant/"

    try:
        logged_states_path = sys.argv[2]
    except:
        logged_states_path = "./logged_states/anttargetpos/"
    autoencoder_trainer = AutoencoderWrapper(AntTargetPosLowLevel(), checkpoint_path=checkpoint_path, logged_states_path=logged_states_path)
    autoencoder_trainer.train()
