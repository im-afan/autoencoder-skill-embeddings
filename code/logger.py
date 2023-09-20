import torch
import numpy as np

class LoggedState:
    def __init__(self, orig_state, end_state, episode):
        self.orig_state = orig_state
        self.end_state = end_state
        self.episode = episode

logged_states = []

def log_state(self, orig_state, end_state, episode):
    logged_states.append(LoggedState(orig_state, end_state, episode))