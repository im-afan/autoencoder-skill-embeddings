import torch
import numpy as np
import pandas as pd

class LoggedState:
    def __init__(self, orig_state, end_state, action, episode):
        self.orig_state = np.squeeze(np.array(orig_state, dtype=np.float32))
        self.end_state = np.squeeze(np.array(end_state, dtype=np.float32))
        self.action = np.squeeze(np.array(action, dtype=np.float32))
        self.episode = episode

logged_states = []
logged_orig_states = []
logged_end_states = []
logged_actions = []

def log_state(orig_state, end_state, action, episode):
    if(type(orig_state) == type((0, 0))):
        return
    logged_states.append(LoggedState(orig_state, end_state, action, episode))
    logged_orig_states.append(orig_state)
    logged_end_states.append(end_state)
    logged_actions.append(action)

def write_logs_to_file(log_path="./logged_states"):
    global logged_states, logged_orig_states, logged_end_states, logged_actions
    df = pd.DataFrame(logged_states)
    for state in logged_states:
        df1 = pd.DataFrame({
            "original_state": [state.orig_state],
            "end_state": [state.end_state],
            "action": [state.action],
            "episode": [state.episode]
        })
        df = pd.concat((df, df1))

    logged_orig_states = np.array(logged_orig_states)
    logged_end_states = np.array(logged_end_states)
    logged_actions = np.array(logged_actions)

    np.save(log_path+"/logged_orig_states", logged_orig_states)
    np.save(log_path+"/logged_end_states", logged_end_states)
    np.save(log_path+"/logged_actions", logged_actions)
    
    df.to_csv(log_path + "/" + "states.csv")
