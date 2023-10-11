import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosBulletEnv, AntTurnHighLevelEnv
from custom_envs.ant_obstacle import AntObstacleLowLevelEnv, AntObstacleHighLevelEnv
from custom_envs.humanoid_turn import HumanoidTargetPosHighLevel, HumanoidTargetPosLowLevel

def register_envs():
    gym.register(
        id="AntTargetPosBulletEnv-v0",
        entry_point=AntTargetPosBulletEnv
    )
    gym.register(
        id="AntTargetPosHighLevel-v0",
        entry_point=AntTurnHighLevelEnv
    )
    gym.register(
        id="AntObstacleLowLevel-v0",
        entry_point=AntObstacleLowLevelEnv
    )
    gym.register(
        id="AntObstacleHighLevel-v0",
        entry_point=AntObstacleHighLevelEnv
    )
    gym.register(
        id="HumanoidTargetPosLowLevel-v0",
        entry_point=HumanoidTargetPosLowLevel
    )
    gym.register(
        id="HumanoidTargetPosHighLevel-v0",
        entry_point=HumanoidTargetPosHighLevel
    )