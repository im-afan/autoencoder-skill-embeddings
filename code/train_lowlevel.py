import sys
from rl_zoo3.train import train
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosLowLevel
from register_envs import register_envs

register_envs()

wanted_env = sys.argv[1]
sys.argv.pop(1);
sys.argv = sys.argv + ["--algo", "ppo", "--env", wanted_env+"LowLevel-v0", "--eval-freq", "100000",
                        "--tensorboard-log", "tensorboard/"+wanted_env+"LowLevel", "--conf-file", "ppo_config.yml", "-n", "750000", "-P"]

train()