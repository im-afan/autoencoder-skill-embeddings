import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosBulletEnv, AntTurnHighLevelEnv

def register_envs():
    gym.register(
        id="AntTargetPosBulletEnv-v0",
        entry_point=AntTargetPosBulletEnv
    )
    gym.register(
        id="AntTargetPosHighLevel-v0",
        entry_point=AntTurnHighLevelEnv
    )