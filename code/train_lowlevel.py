import sys
from rl_zoo3.train import train
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosBulletEnv

gym.register(
    id="AntTargetPosBulletEnv-v0",
    entry_point=AntTargetPosBulletEnv
)
sys.argv = ["python", "--algo", "ppo", "--env", "AntTargetPosBulletEnv-v0", "--eval-freq", "100"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]

train()