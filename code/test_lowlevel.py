import sys
#from rl_zoo3.enjoy import enjoy
from custom_enjoy import enjoy
import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosLowLevel
from logger import write_logs_to_file
from register_envs import register_envs

register_envs()

wanted_env = sys.argv[1]
folder = sys.argv[2]
sys.argv.pop(1)
sys.argv.pop(1)
sys.argv = sys.argv + ["--algo", "ppo", "--env", wanted_env+"LowLevel-v0", "-n", "40000", "--folder", folder+"/agents/"]

enjoy()
write_logs_to_file(log_path=folder+"/logged_states/"+wanted_env.lower())
