import torch 
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from movement_autoencoder import Encoder 
from torch.utils.data import DataLoader, Dataset
import logger
import project_config
import plotly.graph_objects as go

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

encoder = Encoder(0, 36, project_config.AUTOENCODER_LATENT_SIZE_ANT)
encoder.load_state_dict(torch.load("trials/0/autoencoders/encoder.pth"))

running_loss = 0

logged_states_path = "./trials/0/logged_states/anttargetpos/"
logged_orig_states = np.load(logged_states_path+"logged_orig_states.npy")
logged_end_states = np.load(logged_states_path+"logged_end_states.npy")
logged_actions = np.load(logged_states_path+"logged_actions.npy")
for i in range(len(logged_orig_states)):
    logger.log_state(logged_orig_states[i], logged_end_states[i], logged_actions[i], 0)
dataset = MovementDataset(logger.logged_states)

loader = DataLoader(dataset, batch_size=16)

x, y, z = [], [], []

for batch_index, (begin_state, end_state, action) in enumerate(loader):
    #print("index: {}, orig: {}, end: {}".format(batch_index, begin_state.shape, end_state.shape))
    #print(action)
    output = encoder(begin_state, action).detach().numpy()
    print(output.shape)
    for i in range(len(output)):
        #ax.scatter(output[i][0], output[i][1], output[i][2], c=(0, 0, 0, (action[i][0] + 2)/4))
        #ax.scatter(output[i][0], output[i][1], c=(0, 0, 0, (action[i][0] + 2)/4))
        x.append(output[i][0])
        y.append(output[i][1])
        z.append(output[i][2])
    print(batch_index)

markers = go.Scatter3d(x=x, y=y, z=z, marker=go.scatter3d.Marker(size=1), opacity=0.5, mode='markers')
fig = go.Figure(data=markers)
fig.show()
