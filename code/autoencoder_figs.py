import hiddenlayer as hl
from torchviz import make_dot
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logger
from movement_autoencoder import Autoencoder, Encoder, Decoder
import project_config

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

logged_states_path = "trials/1/logged_states/anttargetpos/"
logged_orig_states = np.load(logged_states_path+"logged_orig_states.npy")
logged_end_states = np.load(logged_states_path+"logged_end_states.npy")
logged_actions = np.load(logged_states_path+"logged_actions.npy")
for i in range(len(logged_orig_states)):
    logger.log_state(logged_orig_states[i], logged_end_states[i], logged_actions[i], 0)
dataset = MovementDataset(logger.logged_states)


model = Autoencoder(28, 8, project_config.AUTOENCODER_LATENT_SIZE_ANT)
loader = DataLoader(dataset, batch_size=16)

for batch_index, (begin_state, end_state, action) in enumerate(loader):
    """             #print("index: {}, orig: {}, end: {}".format(batch_index, begin_state.shape, end_state.shape))
                optimizer.zero_grad()

                output = self.autoencoder(begin_state, action)

                loss = loss_fn(output, action)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
    """
    #print(model(begin_state, action))
    
    transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
    graph = hl.build_graph(model, (begin_state, action), transforms=transforms)
    graph.theme = hl.graph.THEMES['blue'].copy()
    graph.save('rnn_hiddenlayer', format='png')
