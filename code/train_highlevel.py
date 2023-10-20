from register_envs import register_envs
from custom_enjoy import enjoy
from rl_zoo3.train import train
import sys

register_envs()

wanted_env = sys.argv[1]
sys.argv.pop(1);
sys.argv = sys.argv + ["--algo", "ppo", "--env", wanted_env+"HighLevel-v0", "--eval-freq", "100000",
                        "--tensorboard-log", "tensorboard/"+wanted_env+"HighLevel", "-n", "1000000", "--conf-file", "ppo_config.yml", "-P"]

train()