import torch
import numpy as np

class LoggedState:
    def __init__(self, orig_state, end_state, action, episode):
        self.orig_state = np.array(orig_state, dtype=np.float32)
        self.end_state = np.array(end_state, dtype=np.float32)
        self.action = np.array(action, dtype=np.float32)
        self.episode = episode

logged_states = []

def log_state(orig_state, end_state, action, episode):
    if(type(orig_state) == type((0, 0))):
        return
    logged_states.append(LoggedState(orig_state, end_state, action, episode))