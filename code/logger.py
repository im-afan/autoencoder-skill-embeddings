import torch
import numpy as np
import pandas as pd

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

def write_logs_to_file(log_path="code/logged_states/states.csv"):
    df = pd.DataFrame({"original_state": [], "end_state": [], "action": [], "episode": []})
    for state in logged_states:
        df1 = pd.DataFrame({
            "original_state": [state.orig_state],
            "end_state": [state.end_state],
            "action": [state.action],
            "episode": [state.episode]
        })
        df = pd.concat((df, df1))
    df.to_csv(log_path)
