import os
import pybullet_data
import pybullet
from pybullet_envs_gymnasium.robot_locomotors import WalkerBase, Ant
from custom_envs.ant_turn_pybullet import WalkerTargetPosBulletEnv
import torch
from movement_autoencoder import Decoder
import project_config
import numpy as np
from gymnasium.spaces import Box
from pybullet_envs_gymnasium.scene_abstract import Scene

class ObstacleStadiumScene(Scene):
    zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
        if self.stadiumLoaded == 0:
            self.stadiumLoaded = 1

            # stadium_pose = cpp_household.Pose()
            # if self.zero_at_running_strip_start_line:
            # 	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

            filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            filename = os.path.join(pybullet_data.getDataPath(), "cube.urdf")
            self.obstacle_cube_mjcf1 = self._p.loadURDF(filename, basePosition=[0, 3, 1], globalScaling=2)
            self.obstacle_cube_mjcf2 = self._p.loadURDF(filename, basePosition=[0, -4, 1], globalScaling=2)
            self.obstacle_cube_mjcf3 = self._p.loadURDF(filename, basePosition=[4, 0, 1], globalScaling=2)
            self.obstacle_cube_mjcf4 = self._p.loadURDF(filename, basePosition=[-4, 0, 1], globalScaling=2)
            obstacle_list = [self.obstacle_cube_mjcf1, self.obstacle_cube_mjcf2, self.obstacle_cube_mjcf3, self.obstacle_cube_mjcf4]

            # filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
            # self.ground_plane_mjcf = self._p.loadSDF(filename)
            #
            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

            for i in obstacle_list:
                self._p.changeDynamics(i, -1, mass=10000000)



            # 	for j in range(p.getNumJoints(i)):
            # 		self._p.changeDynamics(i,j,lateralFriction=0)
            # despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground


class SinglePlayerStadiumSceneObstacle(ObstacleStadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False


class MultiplayerStadiumSceneObstacle(ObstacleStadiumScene):
    multiplayer = True
    players_count = 3

    def actor_introduce(self, robot):
        ObstacleStadiumScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)

class AntObstacle(WalkerBase):
    foot_list = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]

    def __init__(self):
        WalkerBase.__init__(self, "ant_obstacle.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

class AntObstacleLowLevelEnv(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = Ant()
        scene = SinglePlayerStadiumSceneObstacle
        kwargs["use_target_pos"] = False
        kwargs["custom_scene"] = scene
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render=render, **kwargs)
        #self.robot


class AntObstacleHighLevelEnv(WalkerTargetPosBulletEnv):
    def __init__(self, render=False, **kwargs):
        self.robot = Ant()
        scene = SinglePlayerStadiumSceneObstacle
        kwargs["custom_scene"] = scene
        kwargs["use_target_pos"] = False
        WalkerTargetPosBulletEnv.__init__(self, self.robot, render, **kwargs)
        
        try:
            state_dict = torch.load(kwargs["decoder_path"])
        except:
            state_dict = torch.load("./autoencoder_pretrained/decoder.pth")
        #print(self.action_space.shape[0], self.observation_space.shape[0])
        self.decoder = Decoder(self.observation_space.shape[0],
                               self.action_space.shape[0], 
                               project_config.AUTOENCODER_LATENT_SIZE)
        self.decoder.load_state_dict(state_dict)

        self.action_space = Box(
            np.zeros((project_config.AUTOENCODER_LATENT_SIZE)), 
            np.ones((project_config.AUTOENCODER_LATENT_SIZE))
        )

        #print(self.action_space)

        
    def step(self, a):
        state = torch.tensor(self.robot.calc_state())
        latent = torch.tensor(a)
        #print("a shape: ", a.shape)
        action = self.decoder(state, latent).detach().numpy()
        #action = np.ones_like(action) / 3
        #print(action)
        return super().step(action)



