import torch
import torch.nn as nn
import movement_autoencoder
import torchviz
import matplotlib.pyplot as plt
import numpy as np
from movement_autoencoder import Encoder, Decoder, Autoencoder
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

autoencoder = Autoencoder(28, 8, project_config.AUTOENCODER_LATENT_SIZE_ANT)
autoencoder.encoder.load_state_dict(torch.load("trials/0/autoencoders/encoder.pth"))
autoencoder.decoder.load_state_dict(torch.load("trials/0/autoencoders/decoder.pth"))

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("summaries/")

logged_states_path = "./trials/0/logged_states/anttargetpos/"
logged_orig_states = np.load(logged_states_path+"logged_orig_states.npy")
logged_end_states = np.load(logged_states_path+"logged_end_states.npy")
logged_actions = np.load(logged_states_path+"logged_actions.npy")
for i in range(len(logged_orig_states)):
    logger.log_state(logged_orig_states[i], logged_end_states[i], logged_actions[i], 0)
dataset = MovementDataset(logger.logged_states)

loader = DataLoader(dataset, batch_size=1)

x, y, z = [], [], []

for batch_index, (begin_state, end_state, action) in enumerate(loader):
    #print("index: {}, orig: {}, end: {}".format(batch_index, begin_state.shape, end_state.shape))
    #print(action)
    input_names = ["original_state", "action"]
    output_names = ["reconstructed_action"]
    output = autoencoder(begin_state, action)
    writer.add_graph(autoencoder, (begin_state, action))
    writer.close()

    torch.onnx.export(autoencoder, (begin_state, action), "summaries/autoencoder.onnx", input_names=input_names, output_names=output_names)
    #torchviz.make_dot(output, params=dict(autoencoder.named_parameters())).render("autoencoder", format="png")
    print(output)
    break

#markers = go.Scatter3d(x=x, y=y, z=z, marker=go.scatter3d.Marker(size=1), opacity=0.5, mode='markers')
#fig = go.Figure(data=markers)

