from register_envs import register_envs
from custom_enjoy import enjoy
from rl_zoo3.train import train
import sys

register_envs()

#sys.argv = ["python", "--algo", "ppo", "--env", "AntTargetPosHighLevelEnv-v0", "--folder", "logs/", "-n", "10000"]
sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleHighLevel-v0", "--eval-freq", "100000", "--tensorboard-log", "/tmp/stable-baselines3", "-P"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]

train()