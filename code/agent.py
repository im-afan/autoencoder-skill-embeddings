

import torch
import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.noise import NormalActionNoise
from copy import deepcopy

import project_config
import logger

import time

class Agent:
    def __init__(self, env, save_path="./sb3_pretrained"):
        self.env = env
        action_dim = self.env.action_space.shape[0]
        self.action_noise = NormalActionNoise(
            action_dim,
            sigma=0.1*np.ones(action_dim)
        )
        self.policy = sb3.TD3(
            "MlpPolicy",
            self.env,
            action_noise=self.action_noise,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.save_path = save_path 

    def train(self, total_timesteps, log_interval, save_timesteps):
        print("======= BEGIN TRAINING AGENT =======")
        """for i in range(total_timesteps//save_timesteps):
            print(" asdfasdfadf ")
            #self.policy.load(self.save_path)
            self.policy.learn(
                total_timesteps=save_timesteps, 
                log_interval=log_interval
            )
            print("========== SAVING AGENT =========")
            #self.policy.save(self.save_path)"""
        self.policy.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
        self.policy.save(self.save_path)
        print("======= TRAINING AGENT FINISHED =======")
    
    def sample_movement(self, total_timesteps, render=False):
        print("====== sample movement =======")
        cur_timesteps = 0
        vec_env = self.policy.get_env() 
        #print(vec_env)
        obs = vec_env.reset()
        while cur_timesteps < total_timesteps:
            action, _ = self.policy.predict(obs)
            prev_obs = deepcopy(obs)
            #print(type(vec_env.step(action)))
            obs, _, done, _ = vec_env.step(action)
            logger.log_state(prev_obs, obs, action, cur_timesteps)

            if(render):
                vec_env.render("human")
            if(done):
                print("FINISHED AAAAAASD F")
                vec_env.reset()

            cur_timesteps += 1
            #time.sleep(0.5)
        print("======= finished sampling, writing to file =========")
        logger.write_logs_to_file()
        
