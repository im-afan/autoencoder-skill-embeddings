from register_envs import register_envs
from custom_enjoy import enjoy
from rl_zoo3.train import train
import sys

register_envs()

wanted_env = sys.argv[1]
sys.argv.pop(1);
sys.argv = sys.argv + ["--algo", "ppo", "--env", wanted_env+"HighLevel-v0", "--eval-freq", "100000",
                        "--tensorboard-log", "tensorboard/"+wanted_env+"HighLevel", "-n", "2000000", "--conf-file", "ppo_config.yml", "-P"]

train()

"""
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from register_envs import register_envs
import project_config

register_envs()

wanted_env = sys.argv[1]
sys.argv.pop(1);

env = gym.make(wanted_env+"HighLevel-v0")
model = PPO(
    policy=project_config.policy,
    env=env,
    learning_rate=project_config.learning_rate,
    n_steps=project_config.n_steps,
    batch_size=project_config.batch_size,
    n_epochs=project_config.batch_size,
    gamma=project_config.gamma,
    gae_lambda=project_config.gae_lambda,
    clip_range=project_config.clip_range,
    ent_coef=project_config.ent_coef,
    vf_coef=project_config.vf_coef,
    max_grad_norm=project_config.max_grad_norm,
    use_sde=project_config.use_sde,
    sde_sample_freq=project_config.sde_sample_freq,
    tensorboard_log="tensorboard/"+wanted_env+"HighLevel-v0",
    policy_kwargs=project_config.policy_kwargs,
    verbose=1
)
"""