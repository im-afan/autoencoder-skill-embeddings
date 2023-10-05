import sys
#from rl_zoo3.enjoy import enjoy
from custom_enjoy import enjoy
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosBulletEnv
from logger import write_logs_to_file
from register_envs import register_envs

register_envs()

sys.argv = ["python", "--algo", "ppo", "--env", "AntTargetPosBulletEnv-v0", "--folder", "logs/", "-n", "10000"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleHighLevel-v0", "--folder", "logs/", "-n", "10000"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntObstacleLowLevel-v0", "--folder", "logs/", "-n", "10000"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]

enjoy()
write_logs_to_file()
