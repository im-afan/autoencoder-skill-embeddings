import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, action_size, latent_size, hidden_size=32):
        super().__init__()
        self.dense1 = nn.Linear(2*action_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, latent_size)
        
    def forward(self, state_orig, state_end):
        x = torch.concat(state_orig, state_end);
        x = self.dense1(x);
        x = self.dense2(x);
        return x

class Decoder(nn.Module):
    def __init__(self, action_size, latent_size, hidden_size=32):
        super().__init__()
        self.dense1 = nn.Linear(action_size+latent_size, hidden_size) # takes in original state + embedding
        self.dense2 = nn.Linear(hidden_size, action_size)

    def forward(self, state_orig, latent_encoding):
        x = torch.concat(state_orig, latent_encoding)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, action_size, latent_size, hidden_size=32):
        super().__init__()
        self.encoder = Encoder(action_size, latent_size, hidden_size=hidden_size)
        self.decoder = Encoder(action_size, latent_size, hidden_size=hidden_size)

    def forward(self, state_orig, state_end):
        latent = self.encoder(state_orig, state_end)
        return self.decoder(state_orig, latent)
