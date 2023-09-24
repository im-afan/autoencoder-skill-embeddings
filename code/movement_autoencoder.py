import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, hidden_size=32):
        super().__init__()
        self.dense1 = nn.Linear(observation_size+action_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, latent_size)
        
    def forward(self, state_orig, action):
        x = torch.concat((state_orig, action), dim=1)
        #print(x.shape, self.dense1.in_features)
        x = self.dense1(x);
        x = nn.ReLU()(x)
        x = self.dense2(x);
        x = nn.Sigmoid()(x)
        return x

class Decoder(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, hidden_size=32):
        super().__init__()
        print("wanted shape: ", observation_size+latent_size)
        self.dense1 = nn.Linear(observation_size+latent_size, hidden_size) # takes in original state + embedding
        self.dense2 = nn.Linear(hidden_size, action_size)

    def forward(self, state_orig, latent_encoding):
        if(len(state_orig.shape) == 2):
            x = torch.concat((state_orig, latent_encoding), dim=1)
        else:
            x = torch.concat((state_orig, latent_encoding), dim=0)
        x = self.dense1(x)
        x = nn.ReLU()(x)
        x = self.dense2(x)
        #x = nn.Sigmoid()(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, hidden_size=32):
        super().__init__()
        self.encoder = Encoder(observation_size, action_size, latent_size, hidden_size=hidden_size)
        self.decoder = Decoder(observation_size, action_size, latent_size, hidden_size=hidden_size)

    def save(self, checkpoint_path="Autoencoder_pretrained/"):
        torch.save(self.encoder.state_dict(), checkpoint_path + "encoder.pth")
        torch.save(self.decoder.state_dict(), checkpoint_path + "decoder.pth")

    def forward(self, state_orig, action):
        latent = self.encoder(state_orig, action)
        return self.decoder(state_orig, latent)
