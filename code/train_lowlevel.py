import sys
from rl_zoo3.train import train
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosLowLevel
from register_envs import register_envs

register_envs()

wanted_env = sys.argv[1]
folder = sys.argv[2]
sys.argv.pop(1)
sys.argv.pop(1)

#sys.argv = sys.argv + ["--algo", "ppo", "--env", wanted_env+"LowLevel-v0", "--eval-freq", "100000", "--log-folder", folder + "/agents",
#                        "--tensorboard-log", "tensorboard/"+wanted_env+"LowLevel", "--conf-file", "ppo_config.yml", "-n", "2000000", "-P"]
sys.argv = sys.argv + ["--algo", "ppo", "--env", wanted_env, "--eval-freq", "100000", "--log-folder", folder + "/agents",
                        "--tensorboard-log", "tensorboard/"+wanted_env, "--conf-file", "ppo_config.yml", "-n", "2000000", "-P"]



train()