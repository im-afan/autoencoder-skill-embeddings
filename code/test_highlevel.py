import sys
#from rl_zoo3.enjoy import enjoy
from custom_enjoy import enjoy
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosLowLevel
from logger import write_logs_to_file
from register_envs import register_envs


register_envs()

wanted_env = sys.argv[1]
test_path = sys.argv[2]
sys.argv.pop()
sys.argv.pop()

with open("cur_path.txt", "w") as f:
    f.write(test_path)
#sys.argv = ["python", "--algo", "ppo", "--env", "AntTargetPosHighLevel-v0", "--folder", "logs/", "-n", "10000"]
sys.argv = ["python", "--algo", "ppo", "--env", wanted_env + "HighLevel-v0", "--folder", test_path + "/agents/", "-n", "10000"]
#sys.argv = ["python", "--algo", "ppo", "--env", "HumanoidTargetPosHighLevel-v0", "--folder", "logs/", "-n", "10000"]
#sys.argv = ["python", "--algo", "ppo", "--env", "AntBulletEnv-v0"]

enjoy()
#write_logs_to_file()

