from register_envs import register_envs
from custom_enjoy import enjoy
from rl_zoo3.train import train
import sys

register_envs()

#sys.argv = ["python", "--algo", "ppo", "--env", "AntTargetPosHighLevel-v0", "--eval-freq", "100000", "--tensorboard-log", "/tmp/stable-baselines3", "-i", "logs/ppo/AntTargetPosHighLevel-v0_43/AntTargetPosHighLevel-v0.zip", "-P"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleHighLevel-v0", "--eval-freq", "100000", "--tensorboard-log", "/tmp/stable-baselines3", "-P"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]
sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleHighLevel-v0", "--eval-freq", "100000", "-i", "logs/ppo/AntObstacleHighLevel-v0_1/AntObstacleHighLevel-v0.zip", "--tensorboard-log", "/tmp/stable-baselines3", "-P"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleHighLevel-v0", "--eval-freq", "100000", "--tensorboard-log", "/tmp/stable-baselines3", "-P"]

train()