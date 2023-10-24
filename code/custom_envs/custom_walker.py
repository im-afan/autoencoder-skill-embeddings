from math import sqrt
import pybullet
import numpy as np
import pybullet_envs_gymnasium
from pybullet_envs_gymnasium.env_bases import MJCFBaseBulletEnv
from pybullet_envs_gymnasium.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs_gymnasium.robot_locomotors import WalkerBase, Humanoid
from pybullet_envs_gymnasium.gym_locomotion_envs import WalkerBaseBulletEnv
import project_config
from logger import log_state
import time
import torch
from torch import nn
from gymnasium.spaces import Box
from movement_autoencoder import Decoder
from pybullet_envs_gymnasium.scene_stadium import StadiumScene

class WalkerTargetPosBulletEnv(
    MJCFBaseBulletEnv
):  # literally just added like 10 lines to the original impl lol
    def __init__(self, robot: WalkerBase, custom_scene_=None, render=False, use_obstacles=False, target_dist=1e3, **kwargs):
        self.camera_x = 0
        self.walk_target_x = target_dist  # kilometer away
        self.walk_target_y = 0
        self.stateId = -1
        self.target_dist = target_dist
        self.target = 0
        self.cur_time = 0
        self.has_obstacles = False
        self.target_velocity = 0
        self.obstacle_potential = 0

        try:
            self.use_target_pos = kwargs["use_target_pos"]
        except:
            self.use_target_pos = True

        self.scene_class = SinglePlayerStadiumScene 

        self.use_target_velocity = False
        if("target_velocity" in kwargs):
            self.min_target_dist = 5
            self.max_target_dist = 20
            self.use_target_velocity = kwargs["target_velocity"]

        if("custom_scene" in kwargs):
            self.scene_class = kwargs["custom_scene"] 
            self.has_obstacles = True

        try:
            self.logging = kwargs["logging"]
        except:
            self.logging = False

        try:
            render_mode = kwargs["render_mode"]
        except:
            render_mode = "rgb_array"

        MJCFBaseBulletEnv.__init__(self, robot, render)
        self.observation_space = self.observation_space
        self.action_space = self.action_space


    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = self.scene_class(
            bullet_client, gravity=9.8, timestep=0.0165 / 4, frame_skip=4
        )
        return self.stadium_scene

    def reset(self, **kwargs):
        self.cur_time = 0
        angle = np.random.uniform(0, np.pi)
        target_dist = self.target_dist
        if(self.use_target_velocity):
            self.target_dist = np.random.uniform(self.min_target_dist, self.max_target_dist)
        self.target_velocity = target_dist/1000
        self.walk_target_x = np.cos(angle) * self.target_dist
        self.walk_target_y = np.sin(angle) * self.target_dist
        # print("angle: {}, cos: {}, sin: {}".format(angle, np.cos(angle), np.sin(angle)))

        self.robot.walk_target_x = self.walk_target_x
        self.robot.walk_target_y = self.walk_target_y

        if self.stateId >= 0:
            # print("restoreState self.stateId:",self.stateId)
            self._p.restoreState(self.stateId)

        r = MJCFBaseBulletEnv.reset(self)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        floor = self.stadium_scene.ground_plane_mjcf
        try:
            cube = self.stadium_scene.obstacle_cube_mjcf
            self.robot.addToScene(self._p, cube)
        except:
            pass
        (
            self.parts,
            self.jdict,
            self.ordered_joints,
            self.robot_body,
        ) = self.robot.addToScene(self._p, floor)
        #(self.parts, self.jdict, self.ordered_joints, self.robot_body) = (self.robot.parts, self.robot.jdict, self.robot.ordered_joints, self.robot.robot_body) 
        self.ground_ids = set(
            [
                (
                    self.parts[f].bodies[self.parts[f].bodyIndex],
                    self.parts[f].bodyPartIndex,
                )
                for f in self.foot_ground_object_names
            ]
        )
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
            # print("saving state self.stateId:",self.stateId)

        return r

    def _isDone(self):
        return self._alive < 0 or self.cur_time >= 1000

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(
            init_x, init_y, init_z
        )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost = (
        -2.0
    )  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = (
        -0.1
    )  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = (
        -1.0
    )  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1  # discourage stuck joints

    def step(self, a):
        self.cur_time += 1
        # print(self.cur_time)
        # print("step step step")
        orig_state = self.robot.calc_state()

        if (
            not self.scene.multiplayer
        ):  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        if self.logging:
            log_state(orig_state, state, a, self.cur_time)

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z, self.robot.body_rpy[1]
            )
        )  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        if(self.use_target_pos):
            self.potential = self.robot.calc_potential()
        else:
            pos_x, pos_y, pos_z = self.robot.body_xyz
            #dist = abs(pos_x)**1.25 + abs(pos_y)**1.25
            #diff = diff ** 1/1.25
            dist = sqrt(pos_x**2 + pos_y**2)
            self.potential = -(self.target_dist-dist)/self.robot.scene.dt
            #print("dist frm origin: {}".format(dist))
        #print(self.potential, potential_old)
        if(self.use_target_velocity):
            progress = -float(abs(self.target_velocity - (self.potential - potential_old)))
        else:
            progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
            self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(
            np.abs(a * self.robot.joint_speeds).mean()
        )  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        """
        prev_obstacle_potential = self.obstacle_potential
        obstacle_progress = 0
        if(self.has_obstacles):
            #todo account for more obstacles
            obstacle_centers = [(4, 0), (-4, 0), (0, 4), (0, -4)]
            min_dist = 1e9
            for obstacle in obstacle_centers:
                dist = np.sqrt((pos_x-obstacle[0])**2 + (pos_y-obstacle[1])**2)
                min_dist = min(min_dist, dist)
            self.obstacle_potential = min_dist/self.scene.dt
            obstacle_progress = self.obstacle_potential-prev_obstacle_potential
        obstacle_progress *= 1/(1+progress)
        """
        parts_index_list = []
        for i in self.robot.parts:
            parts_index_list.append(self.robot.parts[i].bodyPartIndex)
        obstacle_penalty = self.stadium_scene.get_collision_penalty(parts_index_list)
            

        debugmode = 0
        if debugmode:
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)
            print("obstacle penalty") 
            print(obstacle_penalty)
            time.sleep(0.01)

        self.rewards = [
            self._alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
            obstacle_penalty
            #obstacle_progress,
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), False, {}

    def camera_adjust(self):
        x, y, z = self.robot.body_real_xyz

        self.camera_x = x
        self.camera.move_and_look_at(self.camera_x, y, 1.4, x, y, 1.0)
