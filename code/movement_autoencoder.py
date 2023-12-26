import torch
from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, hidden_size=32, embed_state=True):
        super().__init__()
        if(embed_state):
            self.dense1 = nn.Linear(observation_size+action_size, hidden_size)
        else:
            self.dense1 = nn.Linear(action_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, latent_size)
        self.embed_state = embed_state
        print("latent size: ", latent_size)
        
    def forward(self, state_orig, action):
        if(self.embed_state):
            x = torch.concat((state_orig, action), dim=1)
        else:
            x = action
        #print(x.shape, self.dense1.in_features)
        x = self.dense1(x);
        x = nn.ReLU()(x)
        x = self.dense2(x);
        x = nn.ReLU()(x)
        x = self.dense3(x);
        #x = nn.Sigmoid()(x)
        x = nn.Tanh()(x);
        return x

class Decoder(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, hidden_size=32, embed_state=True):
        super().__init__()
        #print("wanted shape: ", observation_size,latent_size)
        if(embed_state):
            self.dense1 = nn.Linear(latent_size+observation_size, hidden_size)
        else:
            self.dense1 = nn.Linear(latent_size, hidden_size) 
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, action_size)
        self.embed_state = embed_state

    def forward(self, state_orig, latent_encoding):
        #print(state_orig.shape, latent_encoding.shape)
        if(self.embed_state):
            if(len(state_orig.shape) == 2):
                x = torch.concat((state_orig, latent_encoding), dim=1)
            else:
                x = torch.concat((state_orig, latent_encoding), dim=0)
        else:
            x = torch.tensor(latent_encoding)
        #x = torch.tensor(latent_encoding)
        #print(x.dtype)
        #print(x)
        x = self.dense1(x)
        x = nn.ReLU()(x)
        x = self.dense2(x)
        x = nn.ReLU()(x)
        x = self.dense3(x);
        #x = nn.Sigmoid()(x)
        x = nn.Tanh()(x);
        return x

class Autoencoder(nn.Module):
    def __init__(self, observation_size, action_size, latent_size, hidden_size=32, in_embed_state=True, latent_embed_state=True):
        super().__init__()
        self.encoder = Encoder(observation_size, action_size, latent_size, hidden_size=hidden_size, embed_state=in_embed_state)
        self.decoder = Decoder(observation_size, action_size, latent_size, hidden_size=hidden_size, embed_state=latent_embed_state)

    def save(self, checkpoint_path="Autoencoder_pretrained/"):
        torch.save(self.encoder.state_dict(), checkpoint_path + "encoder.pth")
        torch.save(self.decoder.state_dict(), checkpoint_path + "decoder.pth")

    def forward(self, state_orig, action):
        latent = self.encoder(state_orig, action)
        #print(latent.shape)
        return self.decoder(state_orig, latent)
