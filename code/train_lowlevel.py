import sys
from rl_zoo3.train import train

sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]

train()