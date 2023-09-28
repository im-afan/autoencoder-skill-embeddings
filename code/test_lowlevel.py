import stable_baselines3 as sb3
from custom_envs.ant_turn import AntTargetPosEnv
from custom_envs.ant_turn1 import CustomAntEnv
from agent import Agent
import gymnasium as gym

#env = gym.make("Ant-v4", render_mode="rgb_array") 
#env = CustomAntEnv(render_mode="rgb_array")
env = AntTargetPosEnv(render_mode="rgb_array", use_contact_forces=True)
agent = Agent(env)
agent.policy = sb3.TD3.load("./sb3_pretrained/low_level1.zip", env=env)
agent.sample_movement(1000, render=True)