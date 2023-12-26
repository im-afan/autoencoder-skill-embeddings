import gymnasium as gym
from custom_envs.ant_turn_pybullet import AntTargetPosLowLevel, AntTargetPosHighLevel, AntTargetPosVelocityHighLevel
from custom_envs.ant_obstacle import AntObstacleLowLevelEnv, AntObstacleHighLevelEnv
from custom_envs.humanoid_turn import HumanoidTargetPosHighLevel, HumanoidTargetPosLowLevel
from custom_envs.panda_tasks import PickAndPlaceLowLevel, PickAndPlaceHighLevel, ReachWithGripperLowLevel, ReachWithGripperHighLevel, PickUpHighLevel, PickUpLowLevel
from panda_gym.envs.panda_tasks import PickAndPlace

def register_envs():
    gym.register(
        id="AntTargetPosLowLevel-v0",
        entry_point=AntTargetPosLowLevel
    )
    gym.register(
        id="AntTargetPosHighLevel-v0",
        entry_point=AntTargetPosHighLevel
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
        id="AntTargetPosVelocityHighLevel-v0",
        entry_point=AntTargetPosVelocityHighLevel
    )
    gym.register(
        id="HumanoidTargetPosLowLevel-v0",
        entry_point=HumanoidTargetPosLowLevel
    )
    gym.register(
        id="HumanoidTargetPosHighLevel-v0",
        entry_point=HumanoidTargetPosHighLevel
    )
    gym.register(
        id="PickAndPlaceLowLevel-v0",
        entry_point=PickAndPlaceLowLevel,
        max_episode_steps=100
    )
    gym.register(
        id="PickAndPlaceHighLevel-v0",
        entry_point=PickAndPlaceHighLevel,
        max_episode_steps=100
    )
    gym.register(
        id="ReachWithGripperLowLevel-v0",
        entry_point=ReachWithGripperLowLevel,
        max_episode_steps=100
    )
    gym.register(
        id="ReachWithGripperHighLevel-v0",
        entry_point=ReachWithGripperLowLevel,
        max_episode_steps=100
    )
    gym.register(
        id="PickUpLowLevel-v0",
        entry_point=PickUpLowLevel,
        max_episode_steps=100
    )
    gym.register(
        id="PickUpHighLevel-v0",
        entry_point=PickUpHighLevel,
        max_episode_steps=100
    )


if(__name__ == "__main__"):
    register_envs()
    #gym.make("ReachWithGripperLowLevel-v0")
    #gym.make("PickAndPlaceLowLevel-v0")
    gym.make(ReachWithGripperHighLevel)