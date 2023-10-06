import sys
from rl_zoo3.train import train
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosBulletEnv
from register_envs import register_envs

register_envs()

#sys.argv = ["python", "--algo", "ppo", "--env", "AntTargetPosBulletEnv-v0", "--eval-freq", "100000", "-P"]
sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleLowLevel-v0", "--eval-freq", "100000", "--tensorboard-log", "tmp/stable-baselines", "-P"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0", "--eval-freq", "1000", "-P"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]

train()