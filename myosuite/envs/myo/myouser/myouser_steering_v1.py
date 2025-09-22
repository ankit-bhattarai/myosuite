# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for MyoUser Arm Steering model."""
import collections
import numpy as np
import interpax

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

from mujoco_playground._src import mjx_env
import scipy  # Several helper functions are only visible under _src
from myosuite.envs.myo.fatigue import CumulativeFatigue
from myosuite.envs.myo.myouser.base import MyoUserBase, BaseEnvConfig
from myosuite.envs.myo.myouser.utils_steering import cross2d, distance_to_tunnel, tunnel_from_nodes, find_body_by_name, spiral_r_middle, to_cartesian, normalise_to_max, spiral_r, normalise, rotate
from myosuite.envs.myo.myouser.steering_law_calculations import calculate_steering_laws
from dataclasses import dataclass, field
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.spatial.transform import Rotation

@dataclass
class MenuSteeringTaskConfig:
    type: str = "menu_0"
    random_start_pos: bool = False
    distance_reach_metric_coefficient: float = 10.
    screen_distance_x: float = 0.5
    screen_friction: float = 0.1
    ee_name: str = "fingertip"
    obs_keys: List[str] = field(default_factory=lambda: ['qpos', 'qvel', 'qacc', 'fingertip', 'act'])
    omni_keys: List[str] = field(default_factory=lambda: ['screen_pos', 'completed_phase_0_arr', 'start_pos', 'path_percentage', 'distance_to_left_tunnel_bound', 'distance_to_right_tunnel_bound', 'path_angle'])  #TODO: update
    weighted_reward_keys: Dict[str, float] = field(default_factory=lambda: {
        "reach": 0,
        "reach_old": 3.5,
        "bonus_0": 5,
        "bonus_1": 0,
        "bonus_1_old": 20,
        "phase_1_touch": 5,
        "phase_1_tunnel": 0,
        "neural_effort": 0,
        "jac_effort": 1.,
        "power_for_softcons": 15,
        "truncated": -5,
        "truncated_progress": 0,
        "bonus_inside_path": 0,
    })
    max_duration: float = 4.
    max_trials: int = 1
    reset_type: str = "epsilon_uniform"
    rectangle_min_length: float = 0.05
    rectangle_max_length: float = 0.5
    rectangle_min_size: float = 0.03
    rectangle_max_size: float = 0.15
    menu_min_width: float = 0.05
    menu_max_width: float = 0.2
    menu_min_height: float = 0.01
    menu_max_height: float = 0.08
    menu_min_items: int = 2
    menu_max_items: int = 6
    circle_min_radius: float = 0.05
    circle_max_radius: float = 0.2
    circle_min_width: float = 0.05
    circle_max_width: float = 0.25
    circle_sample_points: int = 21
    terminate_out_of_bounds: float = 1.0
    min_dwell_phase_0: float = 0.4
    min_dwell_phase_1: float = 0.15
    tunnel_buffer_size: int = 101
    spiral_start: int = 15
    spiral_end_range: List[int] = field(default_factory=lambda: [9, 11])
    spiral_width_range: List[float] = field(default_factory=lambda: [2, 20])
    spiral_checkpoints: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    spiral_flip: bool = True # If set to true, it flips spiral such that agent starts in the inside and has to reach the outside last.
    spiral_angle_rot: bool = False
    spiral_eval_endings: List[float] = field(default_factory=lambda: [9, 10, 11])
    spiral_eval_widths: List[float] = field(default_factory=lambda: [2, 10, 20])
    sinusoidal_flip: bool = False
    varying_width_points: int = 100
    checkpoint_look_ahead: float = 0.05
    checkpoint_look_behind: float = 0.25

@dataclass
class MenuSteeringEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserMenuSteering"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_menu_steering_myouser.xml"
    task_config: MenuSteeringTaskConfig = field(default_factory=lambda: MenuSteeringTaskConfig())

def default_config() -> config_dict.ConfigDict:
    #TODO: update/make use of env_config parameters!
    env_config = config_dict.create(
        # model_path="myosuite/simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml",
        model_path="myosuite/envs/myo/assets/arm/mobl_arms_index_steering_myouser.xml",
        # model_path="myosuite/envs/myo/assets/arm/mobl_arms_index_myoarm_reaching_myouser.xml",
        # model_path="myosuite/envs/myo/assets/arm/myoarm_reaching_myouser.xml",  #rf"../assets/arm/myoarm_relocate.xml"
        #TODO: use 'wrapper' xml file in assets rather than raw simhive file
        ctrl_dt=0.002 * 25,  # Each control step is 25 physics steps
        sim_dt=0.002,
        vision_mode='',
        vision=config_dict.create(
            # vision_mode="rgbd",
            gpu_id=0,
            render_batch_size=1024,
            num_worlds=1024,
            render_width=120,  #64,
            render_height=120,  #64,
            use_rasterizer=False,
            enabled_geom_groups=[0, 1, 2],
            enabled_cameras=[0],
        ),
        muscle_config=config_dict.create(
            muscle_condition=None,
            sex=None,
            control_type="default",   #"default", "relative"
            noise_params=config_dict.create(
                sigdepnoise_type=None,
                sigdepnoise_level=0.103,
                constantnoise_type=None,
                constantnoise_level=0.185,
            ),
        ),
        
        task_config=config_dict.create(
            distance_reach_metric_coefficient=10.,
            screen_distance_x=0.5,  #0.59
            screen_friction=0.1,
            obs_keys=['qpos', 'qvel', 'qacc', 'fingertip', 'act'], 
            omni_keys=['screen_pos', 'completed_phase_0_arr', 'target'],  #TODO: update!
            weighted_reward_keys=config_dict.create(
                reach=1,
                # bonus_0=0,
                bonus_1=10,
                phase_1_touch=10,
                phase_1_tunnel=10,  #-2,
                neural_effort=0.0,  #1e-4,
                jac_effort=0.05,
                truncated=-10, #0
                truncated_progress=-20,
                bonus_inside_path=0.0,
                
                # ## old reward fct. (florian's branch):
                # reach=1,
                # bonus_0_old=5,
                # bonus_1_old=12,
            ),
            max_duration=4., # timelimit per trial, in seconds
            max_trials=1,  # num of trials per episode
            reset_type="range_uniform",
        ),
        eval_mode=False,
        # episode_length=400,
    )

    rl_config = config_dict.create(
        num_timesteps=15_000_000,  #50_000_000,
        log_training_metrics=True,
        num_evals=0,  #16,
        reward_scaling=0.1,
        # episode_length=env_config.episode_length,
        episode_length=int(env_config.task_config.max_duration / env_config.ctrl_dt),  #TODO: fix, as this dependency is not automatically updated...
        clipping_epsilon=0.3,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=8,  #128,  #32
        num_updates_per_batch=8,  #2,  #8
        num_resets_per_eval=1,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=1024,  #8192,
        batch_size=128,  #512,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 256),
            value_hidden_layer_sizes=(256, 256),
            # policy_obs_key="state",
            # value_obs_key="state",
            # distribution_type="tanh",
        )
    )
    env_config["ppo_config"] = rl_config
    return env_config


class MyoUserMenuSteering(MyoUserBase): 
    def modify_mj_model(self, mj_model):
        default_screen_distance_x = 0.50  #distance from humphant body; corresponds to default screen pos[0] value in xml file

        mj_model.body('screen').pos[0] += self._config.task_config.screen_distance_x - default_screen_distance_x
        mj_model.geom('screen').friction = self._config.task_config.screen_friction
        if any([mj_model.geom(i).name=='fingertip_contact' for i in range(mj_model.ngeom)]):
            mj_model.geom('fingertip_contact').friction = self._config.task_config.screen_friction
        return mj_model
        
    def add_lines(self, spec:mujoco.MjSpec):
        screen = find_body_by_name(spec, "screen")
        if self._config.task_config.type in ["rectangle_0", "menu_0", "menu_1", "menu_2"]:
            n_lines = 12  #TODO: set lower number for "menu_1", "menu_2" and "rectangle_0"
        elif self._config.task_config.type in ["circle_0", "spiral_0", "sinusoidal_0"]:
            circle_sample_points = self._config.task_config.circle_sample_points
            n_lines = 2 * (circle_sample_points - 1)
        elif self._config.task_config.type in ["varying_width"]:
            varying_width_points = self._config.task_config.varying_width_points
            n_lines = 2 * (varying_width_points - 1)
        print(f"Added {n_lines}")
        for i in range(n_lines):
            screen.add_site(pos=[-0.01, 0.25, 0.2], size=[0.001, 0.005, 0.01], name=f"line_{i}", rgba=[0, 0, 0, 0], type=mujoco._enums.mjtGeom(6), euler=[np.pi/2, 0, 0])

    def preprocess_spec(self, spec:mujoco.MjSpec):
        self.add_lines(spec)
        for geom in spec.geoms:
            if (geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER) or (geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        
        return spec    
   
    def _setup(self):
        """Task specific setup"""
        super()._setup()
        self.task_type = self._config.task_config.type
        self.random_start_pos = self._config.task_config.random_start_pos
        self.max_duration = self._config.task_config.max_duration

        # Prepare observation components
        self.obs_keys = self._config.task_config.obs_keys
        self.omni_keys = self._config.task_config.omni_keys
        #TODO: call _prepare_vision() before _setup()?
        if not self._config.vision.enabled:
            print(f"No vision, so adding {self.omni_keys} to obs_keys")
            for key in self.omni_keys:
                if key not in self.obs_keys:
                    self.obs_keys.append(key)
        else:
            print(f"Vision, so not adding {self.omni_keys} to obs_keys")
        print(f"Obs keys: {self.obs_keys}")

        # Prepare reward keys
        self.weighted_reward_keys = self._config.task_config.weighted_reward_keys

        self.screen_id = self._mj_model.site("screen").id
        # self.top_line_id = self._mj_model.site("top_line").id
        # self.bottom_line_id = self._mj_model.site("bottom_line").id
        # self.start_line_id = self._mj_model.site("start_line").id
        # self.end_line_id = self._mj_model.site("end_line").id

        self.ee_name = self._config.task_config.ee_name
        self.fingertip_id = self._mj_model.site(self.ee_name).id

        #self._shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_rot")
        #self._elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        # self.screen_touch_id = self._mj_model.sensor("screen_touch").id

        #TODO: once contact sensors are integrated, check if the fingertip_geom is needed or not

        self.distance_reach_metric_coefficient = self._config.task_config.distance_reach_metric_coefficient

        # Currently hardcoded
        self.rectangle_min_length = self._config.task_config.rectangle_min_length
        self.rectangle_max_length = self._config.task_config.rectangle_max_length
        self.rectangle_min_size = self._config.task_config.rectangle_min_size
        self.rectangle_max_size = self._config.task_config.rectangle_max_size
        self.menu_min_width = self._config.task_config.menu_min_width
        self.menu_max_width = self._config.task_config.menu_max_width
        self.menu_min_height = self._config.task_config.menu_min_height
        self.menu_max_height = self._config.task_config.menu_max_height
        self.menu_min_items = self._config.task_config.menu_min_items
        self.menu_max_items = self._config.task_config.menu_max_items
        self.circle_min_radius = self._config.task_config.circle_min_radius
        self.circle_max_radius = self._config.task_config.circle_max_radius
        self.circle_min_width = self._config.task_config.circle_min_width
        self.circle_max_width = self._config.task_config.circle_max_width
        self.circle_sample_points = self._config.task_config.circle_sample_points
        # self.bottom = self._config.task_config.bottom
        # self.top = self._config.task_config.top
        # self.left = self._config.task_config.left
        # self.right = self._config.task_config.right

        # self.opt_warmstart = True  #self._config.task_config.opt_warmstart
        self.tunnel_buffer_size = self._config.task_config.tunnel_buffer_size
        self.terminate_out_of_bounds = self._config.task_config.terminate_out_of_bounds
        self.min_dwell_phase_0 = self._config.task_config.min_dwell_phase_0
        self.phase_0_completed_min_steps = max(np.ceil(self._config.task_config.min_dwell_phase_0 / self._config.ctrl_dt).astype(int), 1)
        self.min_dwell_phase_1 = self._config.task_config.min_dwell_phase_1
        self.phase_1_completed_min_steps = max(np.ceil(self._config.task_config.min_dwell_phase_1 / self._config.ctrl_dt).astype(int), 1)
        print(f"phase_1_completed_min_steps: {self.phase_1_completed_min_steps}")
        self.non_accumulation_metrics = ['completed_phase_0', 'completed_phase_1', 'dist', 'distance_phase_0', 'distance_phase_1', 'phase_1_x_dist', 'dist_to_start_line', 'dist_to_end_line', 'path_length', 'percentage_achieved'] + [f'crossed_checkpoint_{cp}' for cp in self._config.task_config.spiral_checkpoints]


    # def _prepare_after_init(self, data):
    #     super()._prepare_after_init(data)
        # # Define target origin, relative to which target positions will be generated
        # self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.reach_settings.ref_site)].copy() + jp.array(self.reach_settings.target_origin_rel)  #jp.zeros(3,)

    def get_obs_dict(self, data, info):
        rng = info['rng']

        obs_dict = {}
        obs_dict['time'] = jp.array(data.time)
        
        # Normalise qpos
        jnt_range = self._mj_model.jnt_range[self._independent_joints]
        qpos = data.qpos[self._independent_qpos].copy()
        qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
        qpos = (qpos - 0.5) * 2
        obs_dict['qpos'] = qpos

        # Get qvel, qacc
        obs_dict['qvel'] = data.qvel[self._independent_dofs].copy()  #*self.dt
        obs_dict['qacc'] = data.qacc[self._independent_dofs].copy()

        # Normalise act
        if self._na > 0:
            obs_dict['act']  = (data.act - 0.5) * 2
        
        # Store current control input/qacc state
        obs_dict['last_ctrl'] = data.ctrl.copy()
        # obs_dict['previous_qacc'] = info['qacc'].copy()

        # End-effector and target position - read current position from data instead of cached info
        obs_dict['fingertip'] = data.site_xpos[self.fingertip_id]
        obs_dict['fingertip_past'] = info['fingertip_past']
        obs_dict['screen_pos'] = data.site_xpos[self.screen_id]

        # # # Read positions directly from current data instead of stale info
        # obs_dict['end_line'] = info['end_line']
        # obs_dict['top_line'] = info['top_line']
        # obs_dict['bottom_line'] = info['bottom_line']
        # obs_dict['touching_screen'] = data.sensordata[self.screen_touch_id] > 0.0

        phase_0_completed_steps = info['phase_0_completed_steps']
        phase_1_completed_steps = info['phase_1_completed_steps']
        completed_phase_0 = info['completed_phase_0']
        completed_phase_1 = info['completed_phase_1']
        ee_pos = obs_dict['fingertip']
        current_checkpoint_theta = info["tunnel_checkpoints"][info["current_checkpoint_segment_id"]]
        previous_checkpoint_theta = 0. + (info["current_checkpoint_segment_id"] > 0) * info["tunnel_checkpoints"][info["current_checkpoint_segment_id"] - 1]
        #TODO: move these theta_range values (0.05, -0.25) to config!
        checkpoint_look_ahead = self._config.task_config.checkpoint_look_ahead
        checkpoint_look_behind = self._config.task_config.checkpoint_look_behind
        tunnel_current_theta_max = jp.minimum(current_checkpoint_theta + checkpoint_look_ahead, 1.)  #allow for checkpoint_look_ahead% more, to ensure that checkpoint condition "theta_closest >= current_checkpoint_theta" can be satisfied below
        tunnel_current_theta_min = jp.maximum(previous_checkpoint_theta - checkpoint_look_behind, 0.)  #set lower bound to checkpoint_look_behind% less, as soon as checkpoint condition "theta_closest >= current_checkpoint_theta" has been satisfied below (provides better distance rewards when going the way back)

        # if self.opt_warmstart:
        #     distance_to_tunnel_bounds, theta_closest_left, theta_closest_right = distance_to_tunnel(ee_pos[1:], _interp_fct_left=info['tunnel_boundary_left'], _interp_fct_right=info['tunnel_boundary_right'], theta_init=info['path_percentage'])
        distance_to_left_tunnel_bound, distance_to_right_tunnel_bound, theta_closest_left, theta_closest_right, left_bound_closest, right_bound_closest = distance_to_tunnel(ee_pos[1:], buffer_theta=info['tunnel_boundary_parametrization'], 
                                                                                                                                         buffer_nodes_left=info['tunnel_boundary_left'], buffer_nodes_right=info['tunnel_boundary_right'],
                                                                                                                                         theta_min=tunnel_current_theta_min, theta_max=tunnel_current_theta_max)
        distance_to_tunnel_bounds = jp.minimum(distance_to_left_tunnel_bound, distance_to_right_tunnel_bound)
        obs_dict["left_bound_closest"] = left_bound_closest
        obs_dict["right_bound_closest"] = right_bound_closest

        theta_closest = 0.5 * (theta_closest_left + theta_closest_right)  #TODO: ensure this also works when out of bounds! (e.g., take theta of closer boundary only, depending on task rules)
        obs_dict["path_percentage"] = theta_closest * completed_phase_0
        for cp in self._config.task_config.spiral_checkpoints:
            obs_dict[f'crossed_checkpoint_{cp}'] = (theta_closest >= cp) * completed_phase_0
        
        # start_line = obs_dict['start_line']
        # end_line = obs_dict['end_line']
        # bottom_line_z = obs_dict['bottom_line'][2]
        # top_line_z = obs_dict['top_line'][2]
        # path_length = jp.linalg.norm(end_line[1] - start_line[1])
        # path_width = jp.linalg.norm(top_line_z - bottom_line_z)
        
        nodes = info['tunnel_nodes']
        start_pos = jp.array([obs_dict['screen_pos'][0], *nodes[0]])
        # end_pos = nodes[-1]
        dist_to_start_line = jp.linalg.norm(ee_pos - start_pos, axis=-1)
        node_connections = nodes[1:] - nodes[:-1]
        path_length = jp.sum(jp.linalg.norm(node_connections, axis=1))
        current_node_segment_id = jp.floor(theta_closest * (len(nodes) - 1)).astype(jp.int32)
        current_node_segment_percentage = jp.mod(theta_closest * (len(nodes) - 1), 1)
        # dist_to_end_line = (1. - current_node_segment_percentage) * jp.linalg.norm(node_connections[current_node_segment_id]) + jp.sum(jp.linalg.norm(node_connections[current_node_segment_id+1:], axis=1))
        ## TODO: debug/verify the following line!
        dist_to_end_line = jp.sum(jp.linalg.norm((((1. - current_node_segment_percentage) * (jp.arange(len(node_connections)) == current_node_segment_id)) + 1. * (jp.arange(len(node_connections)) > current_node_segment_id)).reshape(-1, 1) * node_connections, axis=1))
        # dist_to_end_line = jp.linalg.norm(ee_pos[1] - end_pos[1])  #TODO: generalise by measuring 3D distance rather than horizontal distance?
        obs_dict["nodes"] = info['tunnel_nodes']

        # Update phase immediately based on current position
        touching_screen_phase_0 = 1 * (jp.linalg.norm(ee_pos[0] - obs_dict['screen_pos'][0]) <= 0.01)
        close_to_start_pos = 1 * (jp.linalg.norm(ee_pos[1:] - nodes[0]) <= 0.02)
        within_tunnel_limits = distance_to_tunnel_bounds >= 0
        phase_0_completed_now = touching_screen_phase_0 * close_to_start_pos * within_tunnel_limits
        phase_0_completed_steps = (phase_0_completed_steps + 1) * phase_0_completed_now
        completed_phase_0 = completed_phase_0 + (1 - completed_phase_0) * (phase_0_completed_steps >= self.phase_0_completed_min_steps)
        
        phase_1_x_dist = jp.linalg.norm(ee_pos[0] - obs_dict['screen_pos'][0])
        touching_screen_phase_1 = 1 * (phase_1_x_dist <= 0.01)
        inside_tunnel = completed_phase_0 * touching_screen_phase_1 * within_tunnel_limits

        # select next checkpoint segment if checkpoint has been reached
        current_checkpoint_segment_id = (info["current_checkpoint_segment_id"] + 1 * (theta_closest >= current_checkpoint_theta) * inside_tunnel).astype(jp.int32)

        crossed_end_line = theta_closest >= 1
        phase_1_completed_now = inside_tunnel * crossed_end_line
        phase_1_completed_steps = (phase_1_completed_steps + 1) * phase_1_completed_now
        completed_phase_1 = completed_phase_1 + (1 - completed_phase_1) * (phase_1_completed_steps >= self.phase_1_completed_min_steps)   

        current_path_size = jp.linalg.norm(right_bound_closest - left_bound_closest) #TODO: only use this value when inside tunnel? easier way to get path size?
        ## TODO: should we penalize distance to tunnel bounds relative to tunnel size or in absolute terms? 
        softcons_for_bounds = jp.clip(jp.abs(distance_to_tunnel_bounds) / (current_path_size / 2), 0, 1)
        
        # Reset phase 0 when phase 1 is done (and episode ends)
        ## TODO: delay this update to the step function, to ensure consistency between different observations (e.g. when defining the reward function)?
        completed_phase_0 = completed_phase_0 * (1. - completed_phase_1)

        obs_dict["con_0_touching_screen"] = touching_screen_phase_0
        obs_dict["con_0_close_to_start_pos"] = close_to_start_pos
        obs_dict["con_0_1_within_tunnel_limits"] = within_tunnel_limits 
        obs_dict["completed_phase_0"] = completed_phase_0
        obs_dict['completed_phase_0_arr'] = jp.array([completed_phase_0])
        obs_dict["con_1_inside_tunnel"] = inside_tunnel
        obs_dict["con_1_crossed_end_line"] = crossed_end_line
        obs_dict["con_1_touching_screen"] = touching_screen_phase_1
        obs_dict["completed_phase_1"] = completed_phase_1
        obs_dict["softcons_for_bounds"] = softcons_for_bounds
        obs_dict["distance_to_left_tunnel_bound"] = distance_to_left_tunnel_bound
        obs_dict["distance_to_right_tunnel_bound"] = distance_to_right_tunnel_bound
        obs_dict["distance_to_tunnel_bounds"] = distance_to_tunnel_bounds

        ## Compute distances

        phase_0_distance = dist_to_start_line + path_length
        phase_1_distance = dist_to_end_line
        dist = completed_phase_0 * phase_1_distance + (1. - completed_phase_0) * phase_0_distance
        
        obs_dict['dist_to_start_line'] = (1 - completed_phase_0) * dist_to_start_line
        obs_dict['dist_to_end_line'] = dist_to_end_line 
        obs_dict["path_length"] = path_length
        obs_dict["distance_phase_0"] = (1. - completed_phase_0) * phase_0_distance
        obs_dict["distance_phase_1_metric"] = phase_1_distance
        obs_dict["dist"] = dist
        obs_dict["phase_1_x_dist"] = phase_1_x_dist

        ## Additional observations
        ## WARNING: 'target' is only 2D, in contrast to MyoUserSteering (x/depth component is omitted)
        obs_dict['target'] = completed_phase_0 * nodes[1+current_node_segment_id] + (1. - completed_phase_0) * nodes[0]
        obs_dict['start_pos'] = nodes[0]
        obs_dict["path_angle"] = info["tunnel_angle_interp"](theta_closest)
        # obs_dict["completed_phase_0_first"] = (1. - info["completed_phase_0"]) * (obs_dict["completed_phase_0"])
        # obs_dict["completed_phase_1_first"] = (1. - info["completed_phase_1"]) * (obs_dict["completed_phase_1"])
        obs_dict["phase_0_completed_steps"] = phase_0_completed_steps
        obs_dict["phase_1_completed_steps"] = phase_1_completed_steps
        obs_dict["current_node_segment_id"] = current_node_segment_id
        obs_dict["current_checkpoint_segment_id"] = current_checkpoint_segment_id
        obs_dict["remaining_timesteps"] = 1 + jp.round((self.max_duration - data.time)/self._config.ctrl_dt).astype(jp.int32)  #includes current time step

        return obs_dict
    
    def update_info(self, info, obs_dict):
        # TODO: is this really needed? can we drop (almost all) info keys?
        info['last_ctrl'] = obs_dict['last_ctrl']
        # info['qacc'] = obs_dict['qacc']
        # info['motor_act'] = obs_dict['motor_act']
        info['fingertip_past'] = obs_dict['fingertip']
        info["path_percentage"] = obs_dict["path_percentage"]
        info["phase_0_completed_steps"] = obs_dict["phase_0_completed_steps"]
        info["phase_1_completed_steps"] = obs_dict["phase_1_completed_steps"]
        info['completed_phase_0'] = obs_dict['completed_phase_0']
        info['completed_phase_1'] = obs_dict['completed_phase_1']
        info["current_node_segment_id"] = obs_dict["current_node_segment_id"]
        info["current_checkpoint_segment_id"] = obs_dict["current_checkpoint_segment_id"]
        info["remaining_timesteps"] = obs_dict["remaining_timesteps"]

        return info
    
    def get_reward_dict(self, obs_dict):#, info):

        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)

        # Give some intermediate reward for transitioning from phase 0 to phase 1 but only when finger is touching the
        # start line when in phase 0   

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            # ('reach',   1.*(jp.exp(-obs_dict["dist"]*self.distance_reach_metric_coefficient) - 1.)/self.distance_reach_metric_coefficient),  #-1.*reach_dist)
            ('reach',   1.*(1.-(obs_dict['phase_1_completed_steps']>0))*(obs_dict["path_length"] - obs_dict["dist"])/(obs_dict["path_length"])),  #-1.*reach_dist)
            ('reach_old',   -1.*(1.-(obs_dict['phase_1_completed_steps']>0))*obs_dict["dist"]),  #-1.*reach_dist)
            ('bonus_0',   1.*(1.-obs_dict['completed_phase_1'])*((1.-obs_dict['completed_phase_0'])*(obs_dict['con_0_touching_screen'])*(obs_dict['con_0_close_to_start_pos'])*(obs_dict['con_0_1_within_tunnel_limits']))),  #TODO: possible alternative: give one-time bonus when obs_dict['completed_phase_0_first']==True
            ('bonus_1',   1.*(obs_dict['completed_phase_1'])*(obs_dict['remaining_timesteps'])),  #TODO :use obs_dict['completed_phase_1_first'] instead?
            ('bonus_1_old',   1.*(obs_dict['completed_phase_1'])),  #TODO :use obs_dict['completed_phase_1_first'] instead?
            ('phase_1_touch',   1.*(obs_dict['completed_phase_0']*(-obs_dict['phase_1_x_dist']) + (1.-obs_dict['completed_phase_0'])*(-0.3))),
            #('phase_1_touch',   -1.*(obs_dict['completed_phase_0']*(1-obs_dict['con_1_touching_screen']) + (1.-obs_dict['completed_phase_0'])*(0.5))),
            #('phase_1_tunnel', 1.*(1.-obs_dict['completed_phase_1'])*(obs_dict['completed_phase_0']*(-obs_dict['softcons_for_bounds']**15) + (1.-obs_dict['completed_phase_0'])*(-1.))),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            ('jac_effort', -1.* self.get_jac_effort_costs(obs_dict)),
            ('truncated', 1.*(1.-obs_dict["con_0_1_within_tunnel_limits"])*obs_dict["completed_phase_0"]),#jp.logical_or(,(1.0 - obs_dict["con_1_touching_screen"]) * obs_dict["completed_phase_0"])
            ('truncated_progress', 1.*(1.-obs_dict["con_0_1_within_tunnel_limits"])*obs_dict['completed_phase_0']*(1.-obs_dict['path_percentage'])),
            ('bonus_inside_path', 1.*obs_dict['con_1_inside_tunnel']),
            # # Must keys
            ('done',    1.*(obs_dict['completed_phase_1'])), #np.any(reach_dist > far_th))),
        ))

        #TODO: fix this; should go into default dict above
        power_softcons = self.weighted_reward_keys['power_for_softcons']
        phase_1_tunnel_weight = self.weighted_reward_keys['phase_1_tunnel']

        exclude = {"power_for_softcons", "phase_1_tunnel"}

        rwd_dict['dense'] = jp.sum(
            jp.array([wt * rwd_dict[key] for key, wt in self.weighted_reward_keys.items() if key not in exclude]), axis=0) + phase_1_tunnel_weight*(1.*(1.-obs_dict['completed_phase_1'])*(obs_dict['completed_phase_0']*(-obs_dict['softcons_for_bounds']**power_softcons) + (1.-obs_dict['completed_phase_0'])*(-0.3)))

        return rwd_dict
    
    def get_jac_effort_costs(self, obs_dict):
        r_effort = 0.00198*jp.linalg.norm(obs_dict['last_ctrl'])**2 
        r_jacc = 6.67e-6*jp.linalg.norm(obs_dict['qacc'][self._independent_dofs])**2 

        effort_cost = r_effort + r_jacc
        
        return effort_cost
    
    def get_ejk_effort_costs(self, obs_dict):
        r_effort = jp.mean(obs_dict['last_ctrl'])

        qacc = obs_dict['qacc']
        r_jerk = (jp.norm(qacc - obs_dict['previous_qacc']) / self.sim_dt) / 100000.0

        shoulder_ang_vel = obs_dict['qvel'][self._shoulder_id]
        elbow_ang_vel    = obs_dict['qvel'][self._elbow_id]

        shoulder_torque = obs_dict['last_ctrl'][self._shoulder_id]
        elbow_torque    = obs_dict['last_ctrl'][self._elbow_id]

        r_work = (jp.abs(shoulder_ang_vel * shoulder_torque) + jp.abs(elbow_ang_vel * elbow_torque)) / 100.0

        effort_cost = (0.8*r_effort + 6.4*r_jerk + 0.8*r_work) / 8.0

        return effort_cost, qacc

    @staticmethod
    def get_tunnel_limits(rng, low, high, min_size, max_size):
        rng1, rng2 = jax.random.split(rng, 2)
        small_low = low
        small_high = high - min_size
        small_line = jax.random.uniform(rng1) * (small_high - small_low) + small_low
        large_low = small_line + min_size
        large_high = jp.minimum(small_line + max_size, high)
        large_line = jax.random.uniform(rng2) * (large_high - large_low) + large_low

        return small_line, large_line
    
    def get_relevant_positions(self, data: mjx.Data) -> dict[str, jax.Array]:
        return {
            'fingertip': data.site_xpos[self.fingertip_id],
            'screen_pos': data.site_xpos[self.screen_id],
            # 'top_line': data.site_xpos[self.top_line_id],
            # 'bottom_line': data.site_xpos[self.bottom_line_id],
            # 'start_line': data.site_xpos[self.start_line_id],
            # 'end_line': data.site_xpos[self.end_line_id],
        }

    def get_custom_tunnel(self, rng: jax.Array, screen_pos_center: jax.Array,
                          task_type="menu_0", random_start_pos=False,
                          width=None, height=None, n_menu_items=None, tunnel_length=None, circle_radius=None, tunnel_size=None, anchor_pos=None,
                          spiral_end=None, spiral_width=None, phase=None, frequencies=None, tunnel_size_start=None, tunnel_size_end=None) -> dict[str, jax.Array]:
        
        screen_size = self.mj_model.site(self.screen_id).size[1:]
        screen_pos_topleft = screen_pos_center + 0.5*screen_size  #top-left corner of screen is used as anchor [0, 0] in nodes_rel below
        # screen_pos_center_left = screen_pos_center + 0.5*jp.array([screen_size[0], 0])

        if task_type == "rectangle_0":
            screen_margin = jp.array([0.05, 0.05])
            screen_size_with_margin = screen_size - screen_margin  #here, margin is completely used for top-left corner

            min_length, max_length = self.rectangle_min_length, self.rectangle_max_length
            min_size, max_size = self.rectangle_min_size, self.rectangle_max_size

            # Sample tunnel size
            if tunnel_length is None:
                rng, rng_lenth = jax.random.split(rng, 2)
                tunnel_length = min_length + jax.random.uniform(rng_lenth) * (max_length - min_length)
            if tunnel_size is None:
                rng, rng_size = jax.random.split(rng, 2)
                tunnel_size = min_size + jax.random.uniform(rng_size) * (max_size - min_size)

            nodes_rel = jp.array([[0., 0.], [tunnel_length, 0.]])  #nodes_rel are defined in relative coordinates (x, y), with x-axis to the right and y-axis to the top; [0, 0] should correspond to starting point at the top left, so most top-left tunnel boundary point will be [-1.5*width, 0]
            width_height_constraints = None
            total_size = jp.linalg.norm(nodes_rel, axis=0, ord=jp.inf) + 0.5 * jp.array([0, tunnel_size])
            norm_ord = jp.inf  #use max-norm for distance computations/path normalisation

            if anchor_pos is None:
                if random_start_pos:
                    # Sample start pos (centred around screen_pos_topleft)
                    remaining_size = screen_size_with_margin - total_size
                    rng, rng_width_offset = jax.random.split(rng, 2)
                    start_width_offset = jax.random.uniform(rng_width_offset) * remaining_size[0]
                    rng, rng_height_offset = jax.random.split(rng, 2)
                    start_height_offset = jax.random.uniform(rng_height_offset) * remaining_size[1]
                    anchor_pos = screen_pos_topleft - 0.5 * jp.array([0, tunnel_size]) - jp.array([start_width_offset, start_height_offset])
                else:
                    anchor_pos = screen_pos_center + 0.5 * jp.array([tunnel_length, 0])


            tunnel_checkpoints = jp.array([1.0])  #no intermediate checkpoints required for this task, i.e. theta can take any value between 0 and 1 during the entire episode
        
            # Store additional information
            tunnel_extras = {}
        elif task_type == "varying_width":
            screen_margin = jp.array([0.05, 0.05])
            screen_size_with_margin = screen_size - screen_margin  #here, margin is completely used for top-left corner

            min_length, max_length = self.rectangle_min_length, self.rectangle_max_length
            min_size, max_size = self.rectangle_min_size, self.rectangle_max_size

            # Sample tunnel size
            if tunnel_length is None:
                rng, rng_lenth = jax.random.split(rng, 2)
                tunnel_length = min_length + jax.random.uniform(rng_lenth) * (max_length - min_length)
            if (tunnel_size_start is None) or (tunnel_size_end is None):
                rng, rng_size_start, rng_size_end = jax.random.split(rng, 3)
                tunnel_size_start = jax.random.uniform(rng_size_start, minval=2*min_size, maxval=max_size)
                tunnel_size_end = jax.random.uniform(rng_size_end, minval=min_size, maxval=tunnel_size_start)
                tunnel_size = jp.linspace(tunnel_size_start, tunnel_size_end, self._config.task_config.varying_width_points)

            ID = tunnel_length / (tunnel_size_end - tunnel_size_start) * jp.log(tunnel_size_end / tunnel_size_start)

            x_values = jp.linspace(0, tunnel_length, self._config.task_config.varying_width_points)
            y_values = jp.zeros_like(x_values)
            nodes_rel = jp.stack([x_values, y_values], axis=-1)
            width_height_constraints = None
            total_size = jp.linalg.norm(nodes_rel, axis=0, ord=jp.inf) + 0.5 * jp.array([0, tunnel_size_start])
            norm_ord = jp.inf  #use max-norm for distance computations/path normalisation

            if anchor_pos is None:
                if random_start_pos:
                    # Sample start pos (centred around screen_pos_topleft)
                    remaining_size = screen_size_with_margin - total_size
                    rng, rng_width_offset = jax.random.split(rng, 2)
                    start_width_offset = jax.random.uniform(rng_width_offset) * remaining_size[0]
                    rng, rng_height_offset = jax.random.split(rng, 2)
                    start_height_offset = jax.random.uniform(rng_height_offset) * remaining_size[1]
                    anchor_pos = screen_pos_topleft - 0.5 * jp.array([0, tunnel_size]) - jp.array([start_width_offset, start_height_offset])
                else:
                    anchor_pos = screen_pos_center + 0.5 * jp.array([tunnel_length, 0])


            tunnel_checkpoints = jp.array([1.0])  #no intermediate checkpoints required for this task, i.e. theta can take any value between 0 and 1 during the entire episode
        
            # Store additional information
            tunnel_extras = {"ID": ID}   
        elif task_type == "menu_0":
            screen_margin = jp.array([0.05, 0.05])
            screen_size_with_margin = screen_size - screen_margin  #here, margin is completely used for top-left corner

            min_width, max_width = self.menu_min_width, self.menu_max_width
            min_height, max_height = self.menu_min_height, self.menu_max_height

            # Sample tunnel size
            if width is None:
                rng, rng_width = jax.random.split(rng, 2)
                width = min_width + jax.random.uniform(rng_width) * (2/3)*(max_width - min_width)  #default: 0.08
            if height is None:
                rng, rng_height = jax.random.split(rng, 2)
                height = min_height + jax.random.uniform(rng_height) * (max_height - min_height)  #default: 0.05

            nodes_rel = jp.array([[0., 0.], [0., -0.15], [0.2, -0.15], [0.2, -0.3], [0.3, -0.3]])  #nodes_rel are defined in relative coordinates (x, y), with x-axis to the right and y-axis to the top; [0, 0] should correspond to starting point at the top left, so most top-left tunnel boundary point will be [-1.5*width, 0]
            tunnel_size = None #not required, since widths and heights are specified for each node individually using the 'width_height_constraints' arg
            width_height_constraints = [("width", 1.5*width), ("height", height), ("width", width), ("height", height), ("height", height)]
            total_size = jp.linalg.norm(nodes_rel, axis=0, ord=jp.inf) + 0.5 * jp.array([1.5*width, height])
            norm_ord = jp.inf  #use max-norm for distance computations/path normalisation

            if anchor_pos is None:
                if random_start_pos:
                    # Sample start pos (centred around screen_pos_topleft)
                    remaining_size = screen_size_with_margin - total_size
                    rng, rng_width_offset = jax.random.split(rng, 2)
                    start_width_offset = jax.random.uniform(rng_width_offset) * remaining_size[0]
                    rng, rng_height_offset = jax.random.split(rng, 2)
                    start_height_offset = jax.random.uniform(rng_height_offset) * remaining_size[1]
                    anchor_pos = screen_pos_topleft - 0.5 * jp.array([1.5*width, height]) - jp.array([start_width_offset, start_height_offset])
                else:
                    anchor_pos = screen_pos_center + 0.5 * jp.array([0.3, 0.3]) #- 0.5 * jp.array([1.5*width, -height])

            tunnel_checkpoints = jp.array([1.0])  #no intermediate checkpoints required for this task, i.e. theta can take any value between 0 and 1 during the entire episode
        
            # Store additional information
            tunnel_extras = {}
        elif task_type == "menu_1":
            screen_margin = jp.array([0.1, 0.1])
            screen_size_with_margin = screen_size - screen_margin  #here, margin is completely used for top-left corner

            min_width, max_width = self.menu_min_width, self.menu_max_width
            min_height, max_height = self.menu_min_height, self.menu_max_height

            # Sample tunnel size
            if width is None:
                rng, rng_height = jax.random.split(rng, 2)
                width = min_width + jax.random.uniform(rng_width) * (max_width - min_width)  #default: 0.08
            if height is None:
                rng, rng_height = jax.random.split(rng, 2)
                height = min_height + jax.random.uniform(rng_height) * (max_height - min_height)  #default: 0.05

            nodes_rel = jp.array([[0., 0.], [0., -0.15], [0.2, -0.15]])  #nodes_rel are defined in relative coordinates (x, y), with x-axis to the right and y-axis to the top; [0, 0] should correspond to starting point at the top left, so most top-left tunnel boundary point will be [width, 0]
            tunnel_size = None #not required, since widths and heights are specified for each node individually using the 'width_height_constraints' arg
            width_height_constraints = [("width", width), ("height", height), ("height", height)]
            total_size = jp.linalg.norm(nodes_rel, axis=0, ord=jp.inf) + 0.5 * jp.array([width, height])
            norm_ord = jp.inf  #use max-norm for distance computations/path normalisation

            if anchor_pos is None:
                if random_start_pos:
                    # Sample start pos (centred around screen_pos_topleft)
                    remaining_size = screen_size_with_margin - total_size
                    rng, rng_width_offset = jax.random.split(rng, 2)
                    start_width_offset = jax.random.uniform(rng_width_offset) * remaining_size[0]
                    rng, rng_height_offset = jax.random.split(rng, 2)
                    start_height_offset = jax.random.uniform(rng_height_offset) * remaining_size[1]
                    anchor_pos = screen_pos_topleft - 0.5 * jp.array([width, height]) - jp.array([start_width_offset, start_height_offset])
                else:
                    anchor_pos = screen_pos_center + 0.5 * jp.array([0.2, 0.15]) #- 0.5 * jp.array([width, -height])

            tunnel_checkpoints = jp.array([1.0])  #no intermediate checkpoints required for this task, i.e. theta can take any value between 0 and 1 during the entire episode
        
            # Store additional information
            tunnel_extras = {}
        elif task_type == "menu_2":
            screen_margin = jp.array([0.1, 0.075])  #lower screen margin for vertical axis, as we do not display/consider the upper half of the first item, which adds another effective margin in this direction
            screen_size_with_margin = screen_size - screen_margin  #here, margin is completely used for top-left corner

            min_item_width, max_item_width = self.menu_min_width, self.menu_max_width
            min_element_height, max_element_height = self.menu_min_height, self.menu_max_height
            min_n_menu_items, max_n_menu_items = self.menu_min_items, self.menu_max_items

            # Sample element width and height
            if width is None:
                rng, rng_width = jax.random.split(rng, 2)
                width = min_item_width + jax.random.uniform(rng_width) * (max_item_width - min_item_width)  #default: 0.08
            if height is None:
                rng, rng_height = jax.random.split(rng, 2)
                height = min_element_height + jax.random.uniform(rng_height) * (max_element_height - min_element_height)  #default: 0.05
            if n_menu_items is None:
                max_items_permitted = jp.floor(screen_size_with_margin[1] / height).astype(jp.int32)
                rng, rng_n_menu_items = jax.random.split(rng, 2)
                # n_menu_items = jax.random.choice(rng_n_menu_items, jp.arange(min_n_menu_items, jp.minimum(max_n_menu_items, max_items_permitted).astype(jp.int32) + 1))  #default: 4
                n_menu_items = jax.random.choice(rng_n_menu_items, jp.arange(min_n_menu_items, max_n_menu_items + 1))  #default: 4
                n_menu_items = jp.minimum(n_menu_items, max_items_permitted).astype(jp.int32)  #TODO: warning: larger values of n_menu_items might be assigned higher probability, due to clipping after random choice (which is required to avoid dynamic indexing)


            nodes_rel = jp.array([[0., 0.], [0., -(n_menu_items-1)*height], [width, -(n_menu_items-1)*height]])  #nodes_rel are defined in relative coordinates (x, y), with x-axis to the right and y-axis to the top; [0, 0] should correspond to starting point at the top left, so most top-left tunnel boundary point will be [width, 0]
            tunnel_size = None #not required, since widths and heights are specified for each node individually using the 'width_height_constraints' arg
            width_height_constraints = [("width", width), ("height", height), ("height", height)]
            total_size = jp.linalg.norm(nodes_rel, axis=0, ord=jp.inf) + 0.5 * jp.array([width, height])
            norm_ord = jp.inf  #use max-norm for distance computations/path normalisation

            if anchor_pos is None:
                if random_start_pos:
                    # Sample start pos (centred around screen_pos_topleft)
                    remaining_size = screen_size_with_margin - total_size
                    rng, rng_width_offset = jax.random.split(rng, 2)
                    start_width_offset = jax.random.uniform(rng_width_offset) * remaining_size[0]
                    rng, rng_height_offset = jax.random.split(rng, 2)
                    start_height_offset = jax.random.uniform(rng_height_offset) * remaining_size[1]
                    anchor_pos = screen_pos_topleft - 0.5 * jp.array([width, height]) - jp.array([start_width_offset, start_height_offset])
                else:
                    anchor_pos = screen_pos_center + 0.5 * jp.array([width, (n_menu_items-1)*height]) #- 0.5 * jp.array([width, -height])

            tunnel_checkpoints = jp.array([1.0])  #no intermediate checkpoints required for this task, i.e. theta can take any value between 0 and 1 during the entire episode
        
            # Store additional information
            tunnel_extras = {}
        elif task_type == "circle_0":
            screen_margin = jp.array([0.01, 0.01])
            screen_size_with_margin = screen_size - 2*screen_margin  #here, margin is equally distributed between top/bottom and left/right

            min_radius, max_radius = self.circle_min_radius, self.circle_max_radius
            min_width, max_width = self.circle_min_width, self.circle_max_width
            n_sample_points = self.circle_sample_points

            # Sample tunnel length and size
            if circle_radius is None:
                rng, rng_radius = jax.random.split(rng, 2)
                circle_radius = min_radius + jax.random.uniform(rng_radius) * (max_radius - min_radius)  #default: 0.075
                # tunnel_length = circle_radius*(2*jp.pi)
            if tunnel_size is None:
                rng, rng_width = jax.random.split(rng, 2)
                max_width_restricted = jp.minimum(max_width, jp.min(screen_size_with_margin)-2*circle_radius)  #constraint: (2*circle_radius+tunnel_size)<jp.min(screen_size_with_margin)
                tunnel_size = min_width + jax.random.uniform(rng_width) * jp.maximum((max_width_restricted - min_width), 0)  #default: 0.05

            theta_def = np.linspace(0, 1, n_sample_points)
            nodes_rel = circle_radius * np.array([np.sin(theta_def*2*np.pi), np.cos(theta_def*2*np.pi)]).T
            width_height_constraints = None
            total_size = (2 * circle_radius + tunnel_size) * jp.ones(2)
            norm_ord = 2  #use Euclidean norm for distance computations/path normalisation

            if anchor_pos is None:
                if random_start_pos:
                    # Sample start pos (centred around screen_pos_center)
                    remaining_size = screen_size_with_margin - total_size
                    rng, rng_width_offset = jax.random.split(rng, 2)
                    start_width_offset = -0.5*remaining_size[0] + jax.random.uniform(rng_width_offset) * remaining_size[0]
                    rng, rng_height_offset = jax.random.split(rng, 2)
                    start_height_offset = -0.5*remaining_size[1] + jax.random.uniform(rng_height_offset) * remaining_size[1]
                    anchor_pos = screen_pos_center + jp.array([start_width_offset, start_height_offset])
                else:
                    anchor_pos = screen_pos_center

            # tunnel_checkpoints = jp.array([0.5, 1.])  #make sure to reach lower part of circle (theta=0.5) before task can be successfully completed
            tunnel_checkpoints = jp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        
            # Store additional information
            # tunnel_extras = {"tunnel_center": anchor_pos,
            #                  "circle_radius": circle_radius,
            #                  "tunnel_size": tunnel_size}
            tunnel_extras = {}
        elif task_type == "spiral_0":
            n_sample_points = self.circle_sample_points
            start = 15
            end_range = self._config.task_config.spiral_end_range
            width_range = self._config.task_config.spiral_width_range
            flip = self._config.task_config.spiral_flip
            angle_rot = self._config.task_config.spiral_angle_rot
            rng, end_rng, width_rng = jax.random.split(rng, 3)
            if spiral_end is None:
                end = jax.random.uniform(end_rng, minval=end_range[0], maxval=end_range[1])
            else:
                end = spiral_end
            if spiral_width is None:
                width = jax.random.uniform(width_rng, minval=width_range[0], maxval=width_range[1])
            else:
                width = spiral_width
            theta_middle = jp.linspace((start)*jp.pi, (end+2)*jp.pi, n_sample_points)
            r_middle = spiral_r_middle(theta_middle, width)
            x_middle, y_middle = to_cartesian(theta_middle, r_middle)
            x_middle, y_middle, multiplier = normalise_to_max(x_middle, y_middle, maximum=0.3)
            if flip:
                x_middle = jp.flip(x_middle)
                y_middle = jp.flip(y_middle)
            if angle_rot:
                angle = jax.random.uniform(rng, minval=0, maxval=2*jp.pi)
                x_middle, y_middle = rotate(x_middle, y_middle, angle)
            nodes_rel = jp.stack([x_middle, y_middle], axis=-1)
            width_height_constraints = None
            norm_ord = 2
            thetas = jp.linspace(start*jp.pi, end*jp.pi, n_sample_points)
            r_outer = spiral_r(thetas, width)
            r_inner = spiral_r(thetas, width - 2*jp.pi)
            tunnel_size = multiplier * (r_outer - r_inner)
            if flip:
                tunnel_size = jp.flip(tunnel_size)
            if anchor_pos is None:
                if random_start_pos:
                    raise NotImplementedError()
                else:
                    anchor_pos = screen_pos_center
            tunnel_checkpoints = jp.array(self._config.task_config.spiral_checkpoints)
            
            # Store additional information
            tunnel_extras = {
                'r_inner': r_inner,
                'r_outer': r_outer,
                'multiplier': multiplier,
            }
        elif task_type == "sinusoidal_0":
            phase_rng, frequency_rng, flip_rng = jax.random.split(rng, 3)
            n_sample_points = self.circle_sample_points
            components = 3
            x_range = [-0.25, 0.25]
            y_max = 0.25
            if phase is None:
                phase = jax.random.uniform(phase_rng, minval=0, maxval=2*jp.pi, shape=(components,))
            if frequencies is None:
                frequencies = jax.random.uniform(frequency_rng, minval=0.1, maxval=2.25, shape=(components,))
            x = jp.linspace(x_range[0], x_range[1], n_sample_points)
            y = jp.zeros_like(x)
            for i in range(components):
                y += y_max * jp.sin(2*jp.pi*frequencies[i]*(x + phase[i])) / components
            # x, y, _ = normalise_to_max(x, y, y_max)
            nodes_rel = jp.stack([x, y], axis=-1)
            if self._config.task_config.sinusoidal_flip:
                flip = jax.random.choice(flip_rng, a=jp.array([0, 1]), shape=(1,))[0]
                nodes_rel = jax.lax.select(flip, nodes_rel[::-1], nodes_rel)
            width_height_constraints = None
            norm_ord = 2
            tunnel_size = 0.05
            if anchor_pos is None:
                if random_start_pos:
                    raise NotImplementedError()
                else:
                    anchor_pos = screen_pos_center
            tunnel_checkpoints = jp.array(self._config.task_config.spiral_checkpoints)
            
            # Store additional information
            tunnel_extras = {
                'phase': phase,
                'frequencies': frequencies,
            }
        else:
            raise NotImplementedError(f"Task type {task_type} not implemented.")

        nodes = anchor_pos + jp.array([-1., 1.]) * nodes_rel  #map from relative node coordinates (x, y) to MuJoCo coordinates (y, z) using (z <- y, y <- (-x))
        nodes_left, nodes_right, theta_angle, angle = tunnel_from_nodes(nodes, tunnel_size=tunnel_size, width_height_constraints=width_height_constraints, ord=norm_ord)

        # interpolate nodes
        theta = jp.linspace(0, 1, len(nodes))
        # _interp_fct_left = scipy.interpolate.PchipInterpolator(theta, nodes_left, k=1)
        # _interp_fct_right = scipy.interpolate.make_interp_spline(theta, nodes_right, k=1)
        _interp_fct_left = interpax.PchipInterpolator(theta, nodes_left, check=False)
        _interp_fct_right = interpax.PchipInterpolator(theta, nodes_right, check=False)
        
        # discretize interpolated notes for efficient optimization
        buffer_theta = jp.linspace(0, 1, self.tunnel_buffer_size)
        buffer_nodes_left = _interp_fct_left(buffer_theta)
        buffer_nodes_right = _interp_fct_right(buffer_theta)

        # interpolate heading angles
        _interp_fct_angle = interpax.PchipInterpolator(theta_angle, angle, check=False)

        return {'tunnel_nodes': nodes, 
                'tunnel_nodes_left': nodes_left,
                'tunnel_nodes_right': nodes_right,
                'tunnel_boundary_parametrization': buffer_theta,
                'tunnel_boundary_left': buffer_nodes_left, 
                'tunnel_boundary_right': buffer_nodes_right,
                'tunnel_angle': angle,
                'tunnel_angle_theta': theta_angle,
                'tunnel_angle_interp': _interp_fct_angle,  #redundant with (tunnel_angle, tunnel_angle_theta); only stored for convenience and reduced computational workload (but the object type is not compatible with traced arrays, i.e., workarounds are required for domain randomization!)
                'tunnel_checkpoints': tunnel_checkpoints,
                'tunnel_extras': tunnel_extras,
        }

    def get_custom_tunnels_steeringlaw(self, rng: jax.Array, screen_pos_center: jax.Array,
                                       task_type: str ="menu_0", random_start_pos: bool = False, n_tunnels_per_ID: int = 1) -> dict[str, jax.Array]:
        tunnels_total = []
        max_attempts_per_tunnel = 50
        N = 50
        
        if task_type == "rectangle_0":
            for _ in range(N):
                for _ in range(n_tunnels_per_ID):
                    rng, rng2 = jax.random.split(rng, 2)
                    tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type)
                    tunnels_total.append(tunnel_info)
            ## vary lengths for fixed width
            # IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # W = 0.05

            # for ID in IDs:
            #     combos = 0
            #     _attempts = 0
            #     while (combos < n_tunnels_per_ID) and (_attempts < max_attempts_per_tunnel):
            #         rng, rng2 = jax.random.split(rng, 2)
            #         L = ID * W
            #         if self.rectangle_min_length <= L <= self.rectangle_max_length:
            #             anchor_pos = None  #fixed: screen_pos_center; random: None
            #             tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type,
            #                                                  random_start_pos=random_start_pos,
            #                                                  tunnel_length=L, tunnel_size=W,
            #                                                  anchor_pos=anchor_pos)
            #             combos += 1
            #             _attempts = 0
            #             # for i in range(10):
            #             tunnels_total.append(tunnel_info)
            #             print(f"Added path for ID {ID}, L {L}, W {W}")
            #         _attempts +=1
            #     if _attempts == max_attempts_per_tunnel:
            #         print(f"WARNING: Could not find any tunnel of ID {ID} that satisfies the size/width/... constraints from config file.")

            # ## vary widths for fixed length
            # IDs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            # L = 0.5

            # for ID in IDs:
            #     combos = 0
            #     _attempts = 0
            #     while (combos < n_tunnels_per_ID) and (_attempts < max_attempts_per_tunnel):
            #         rng, rng2 = jax.random.split(rng, 2)
            #         W = L/ID
            #         if self.rectangle_min_size <= W <= self.rectangle_max_size:
            #             anchor_pos = None  #fixed: screen_pos_center; random: None
            #             tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type,
            #                                                  random_start_pos=random_start_pos,
            #                                                  width=L, height=W,
            #                                                  anchor_pos=anchor_pos)
            #             combos += 1
            #             _attempts = 0
            #             # for i in range(10):
            #             tunnels_total.append(tunnel_info)
            #             print(f"Added path for ID {ID}, L {L}, W {W}")
            #         _attempts +=1
            #     if _attempts == max_attempts_per_tunnel:
            #         print(f"WARNING: Could not find any tunnel of ID {ID} that satisfies the size/width/... constraints from config file.")
            # IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            # for ID in IDs:
            #     combos = 0
            #     _attempts = 0
            #     while (combos < n_tunnels_per_ID) and (_attempts < max_attempts_per_tunnel):
            #         rng, rng2 = jax.random.split(rng, 2)
            #         W = jax.random.uniform(rng, minval=self.rectangle_min_size, maxval=self.rectangle_max_size)
            #         L = ID * W
            #         if self.rectangle_min_length <= L <= self.rectangle_max_length:
            #             # anchor_pos = None
            #             tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type,
            #                                                   random_start_pos=random_start_pos,
            #                                                   tunnel_length=L, tunnel_size=W)
            #             combos += 1
            #             _attempts = 0
            #             # for i in range(10):
            #             tunnels_total.append(tunnel_info)
            #             print(f"Added path for ID {ID}, L {L}, W {W}")
            #         _attempts +=1
            #     if _attempts == max_attempts_per_tunnel:
            #         print(f"WARNING: Could not find any tunnel of ID {ID} that satisfies the size/width/... constraints from config file.")
        elif task_type == "menu_2":
            for _ in range(N):
                for _ in range(n_tunnels_per_ID):
                    rng, rng2 = jax.random.split(rng, 2)
                    tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type)
                    tunnels_total.append(tunnel_info)
            # IDs = [3, 4, 5, 6, 7, 8, 9, 10]
            
            # # TODO: ensure that same values as in self.get_custom_tunnel() are used!
            # screen_size = self.mj_model.site(self.screen_id).size[1:]
            # screen_margin = jp.array([0.1, 0.075])  #lower screen margin for vertical axis, as we do not display/consider the upper half of the first item, which adds another effective margin in this direction
            # screen_size_with_margin = screen_size - screen_margin  #here, margin is completely used for top-left corner

            # for ID in IDs:
            #     combos = 0
            #     _attempts = 0
            #     while (combos < n_tunnels_per_ID) and (_attempts < max_attempts_per_tunnel):
            #         rng, rng2 = jax.random.split(rng, 2)
            #         rng, rng_height = jax.random.split(rng, 2)
            #         H = jax.random.uniform(rng_height, minval=self.menu_min_height, maxval=self.menu_max_height)
            #         max_items_permitted = jp.floor(screen_size_with_margin[1] / H).astype(jp.int32)
            #         rng, rng_n_menu_items = jax.random.split(rng, 2)
            #         # n_menu_items = jax.random.choice(rng_n_menu_items, jp.arange(min_n_menu_items, jp.minimum(max_n_menu_items, max_items_permitted).astype(jp.int32) + 1))  #default: 4
            #         n_menu_items = jax.random.choice(rng_n_menu_items, jp.arange(self.menu_min_items, self.menu_max_items + 1))  #default: 4
            #         n_menu_items = jp.minimum(n_menu_items, max_items_permitted).astype(jp.int32)  #TODO: warning: larger values of n_menu_items might be assigned higher probability, due to clipping after random choice (which is required to avoid dynamic indexing)

            #         if (n_menu_items < ID**2 / 4):  #otherwise we cannot find an appropriate height!
            #             W = H * (ID + jp.sqrt(ID**2 - 4 * n_menu_items)) / 2  #based on Accot and Zhai menu steering law!
            #             if (self.menu_min_width <= W <= self.menu_max_width):
            #                 # anchor_pos = None
            #                 tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type,
            #                                                     random_start_pos=random_start_pos,
            #                                                     width=W, height=H, n_menu_items=n_menu_items)
            #                 combos += 1
            #                 _attempts = 0
            #                 # for i in range(10):
            #                 tunnels_total.append(tunnel_info)
            #                 print(f"Added path for ID {ID}, W {W}, H {H}, N ITEMS {n_menu_items}")
            #         _attempts +=1
            #     if _attempts == max_attempts_per_tunnel:
            #         print(f"WARNING: Could not find any tunnel of ID {ID} that satisfies the size/width/... constraints from config file.")
        elif task_type == "circle_0":
            ## VARIANT A: fixed IDs, fixed length per ID (for n_tunnels_per_ID=1)
            # IDs = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

            # for ID in IDs:
            #     combos = 0
            #     _attempts = 0
            #     while (combos < n_tunnels_per_ID) and (_attempts < max_attempts_per_tunnel):
            #         rng, rng2 = jax.random.split(rng, 2)
            #         W = jax.random.uniform(rng, minval=self.circle_min_width, maxval=self.circle_max_width)
            #         L = ID * W
            #         circle_radius = (L) / (2 * jp.pi)
            #         if self.circle_min_radius <= circle_radius <= self.circle_max_radius:
            #             # anchor_pos = screen_pos_center
            #             tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type,
            #                                                  random_start_pos=random_start_pos,
            #                                                  circle_radius=circle_radius, tunnel_size=W)
            #             combos += 1
            #             _attempts = 0
            #             # for i in range(10):
            #             tunnels_total.append(tunnel_info)
            #             print(f"Added path for ID {ID}, L {L}, W {W}")
            #         _attempts +=1
            #     if _attempts == max_attempts_per_tunnel:
            #         print(f"WARNING: Could not find any tunnel of ID {ID} that satisfies the size/width/... constraints from config file.")

            # ## VARIANT B: totally random tunnels, as during training
            for _ in range(N):
                for _ in range(n_tunnels_per_ID):
                    rng, rng2 = jax.random.split(rng, 2)
                    tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type)
                    tunnels_total.append(tunnel_info)

            # ## VARIANT C: random IDs and widths
            # N = 250
            # ID_range = [5, 23]
            # rng, rng2 = jax.random.split(rng, 2)
            # IDs = jax.random.uniform(rng2, shape=(N,), minval=ID_range[0], maxval=ID_range[1])

            # for ID in IDs:
            #     combos = 0
            #     _attempts = 0
            #     while (combos < n_tunnels_per_ID) and (_attempts < max_attempts_per_tunnel):
            #         rng, rng2 = jax.random.split(rng, 2)
            #         W = jax.random.uniform(rng, minval=self.circle_min_width, maxval=self.circle_max_width)
            #         L = ID * W
            #         circle_radius = (L) / (2 * jp.pi)
            #         if self.circle_min_radius + 0.0 * (self.circle_max_radius - self.circle_min_radius) <= circle_radius <= self.circle_max_radius - 0.0 * (self.circle_max_radius - self.circle_min_radius):
            #             # anchor_pos = screen_pos_center
            #             tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type,
            #                                                  random_start_pos=random_start_pos,
            #                                                  circle_radius=circle_radius, tunnel_size=W)
            #             combos += 1
            #             _attempts = 0
            #             # for i in range(10):
            #             tunnels_total.append(tunnel_info)
            #             print(f"Added path for ID {ID}, L {L}, W {W}")
            #         _attempts +=1
            #     if _attempts == max_attempts_per_tunnel:
            #         print(f"WARNING: Could not find any tunnel of ID {ID} that satisfies the size/width/... constraints from config file.")

        #print(f"tunnels_total", tunnels_total)
        elif task_type == "spiral_0":
            for _ in range(N):
                for _ in range(n_tunnels_per_ID):
                    rng, rng2 = jax.random.split(rng, 2)
                    tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type)
                    tunnels_total.append(tunnel_info)
            # spiral_endings = self._config.task_config.spiral_eval_endings
            # spiral_widths = self._config.task_config.spiral_eval_widths

            # for spiral_end in spiral_endings:
            #     for spiral_width in spiral_widths:
            #         for _ in range(n_tunnels_per_ID):
            #             rng, rng2 = jax.random.split(rng, 2)
            #             # anchor_pos = screen_pos_center
            #             tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type,
            #                                                  random_start_pos=random_start_pos,
            #                                                  spiral_end=spiral_end, spiral_width=spiral_width)
            #             tunnels_total.append(tunnel_info)
            #             print(f"Added path for spiral_end {spiral_end}, spiral_width {spiral_width}")

        elif task_type == "sinusoidal_0":
            for _ in range(N):
                for _ in range(n_tunnels_per_ID):
                    rng, rng2 = jax.random.split(rng, 2)
                    tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type)
                    tunnels_total.append(tunnel_info)

        elif task_type == "varying_width":
            #TODO: add proper eval code here
            for _ in range(N):
                for _ in range(n_tunnels_per_ID):
                    rng, rng2 = jax.random.split(rng, 2)
                    tunnel_info = self.get_custom_tunnel(rng2, screen_pos_center=screen_pos_center, task_type=task_type)
                    tunnels_total.append(tunnel_info)
        
        print(f"Added {len(tunnels_total)} tunnels in total.")

        return tunnels_total

    def reset(self, rng, tunnel_infos=None, render_token=None):
        # jax.debug.print(f"RESET INIT")

        _, rng = jax.random.split(rng, 2)

        # Reset biomechanical model
        data = self._reset_bm_model(rng)
        screen_pos_center = data.site_xpos[self.screen_id][1:]

        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)
        # qacc = jp.zeros(len(self._independent_dofs))

        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                # "qacc": qacc,
                "fingertip": data.site_xpos[self.fingertip_id],
                "fingertip_past": data.site_xpos[self.fingertip_id],
                "path_percentage": 0.0,
                "phase_0_completed_steps": 0,
                "phase_1_completed_steps": 0,
                "completed_phase_0": 0.0,
                "completed_phase_1": 0.0,
                "current_node_segment_id": 0,
                "current_checkpoint_segment_id": 0,
                "remaining_timesteps": 1 + jp.round((self.max_duration - data.time)/self._config.ctrl_dt).astype(jp.int32),  #includes current time step
                }
        if tunnel_infos is None:
            info.update(self.get_custom_tunnel(rng, screen_pos_center=screen_pos_center, task_type=self.task_type, random_start_pos=self.random_start_pos))
        else:
            info.update(tunnel_infos)

        # Generate inital observations
        # TODO: move the following lines into MyoUserBase.reset?
        if self.vision:
            # TODO: do we need to update target information for rendering?
            # data = self.add_target_pos_to_data(data, info["target_pos"])
            info.update(self.generate_pixels(data, render_token=render_token))
        obs, info = self.get_obs_vec(data, info)  #update info from observation made

        reward, done = jp.zeros(2)
        metrics = {
            'completed_phase_0': 0.0,
            'completed_phase_1': 0.0,
            'dist': 0.0,
            'distance_phase_0': 0.0,
            'distance_phase_1': 0.0,
            'phase_1_x_dist': 0.0,
            'dist_to_start_line': 0.0,
            'dist_to_end_line': 0.0,
            'path_length': 0.0,
            #'con_0_close_to_start_pos: 0.0,
            #'con_0_touching_screen': 0.0,
            #'con_1_touching_screen': 0.0,
            #'con_1_crossed_line_y': 0.0,
            'percentage_achieved': 0.0,
            'distance_to_tunnel_bounds': 0.0,
            'softcons_for_bounds': 0.0,
            'out_of_bounds': 0.0,
            'jac_effort_reward': 0.0,
            #'neural_effort_reward': 0.0,
            'distance_reward': 0.0,
            'bonus_reward': 0.0,
            'touch_reward': 0.0,
            'tunnel_reward': 0.0,
            'not_touching': 0.0,
        }
        for cp in self._config.task_config.spiral_checkpoints:
            metrics[f'crossed_checkpoint_{cp}'] = 0.0
        
        return State(data, obs, reward, done, metrics, info)
    
    def auto_reset(self, rng, info_before_reset, **kwargs):
        render_token = info_before_reset["render_token"] if self.vision else None
        return self.reset(rng, render_token=render_token, **kwargs)
    
    def eval_reset(self, rng, eval_id, **kwargs):
        """Reset function wrapper called by evaluate_policy."""
        # tunnel_infos = jax.tree.map(lambda x: x.at[eval_id].get(), self.SL_tunnel_infos)
        tunnel_infos = {}

        for k in self.SL_tunnel_infos_keys:
            tunnel_infos[k] = self.SL_tunnel_infos[k].at[eval_id].get()
        tunnel_extras = {}
        for k in self.SL_tunnel_infos_keys_extra:
            _val = self.SL_tunnel_infos[k].at[eval_id].get()
            # if not jp.all(jp.isnan(_val)):
            ##NOTE: a single jp.nan value is used for eval_ids that lack a specific key in tunnel_extras
            ##UPDATED NOTE: we need to set all keys at any time due to traced arrays, so nvm...
            tunnel_extras[k] = _val
        tunnel_infos["tunnel_extras"] = tunnel_extras

        if ("tunnel_angle_theta" in self.SL_tunnel_infos_keys) and ("tunnel_angle" in self.SL_tunnel_infos_keys):
            tunnel_infos["tunnel_angle_interp"] = interpax.PchipInterpolator(tunnel_infos["tunnel_angle_theta"], tunnel_infos["tunnel_angle"], check=False)

        # _tunnel_infos_jp = self.SL_tunnel_infos.at[eval_id].get()
        # tunnel_infos = {}
        # tunnel_infos['bottom_line'] = _tunnel_position_jp[0]
        # tunnel_infos['top_line'] = _tunnel_position_jp[1]
        # tunnel_infos['start_line'] = _tunnel_position_jp[2]
        # tunnel_infos['end_line'] = _tunnel_position_jp[3]
        # tunnel_infos['screen_pos'] = _tunnel_position_jp[4]

        return self.reset(rng, tunnel_infos=tunnel_infos, **kwargs)
    
    def prepare_eval_rollout(self, rng, **kwargs):
        """Function that can be used to define random parameters to be used across multiple evaluation rollouts/resets.
        May return the number of evaluation episodes that should be rolled out (before this method should be called again)."""
        
        ## Setup evaluation episodes for Steering Law validation
        rng, rng2 = jax.random.split(rng, 2)
        # self.SL_tunnel_infos = jp.array(self.get_custom_tunnels_different_lengths(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])))
        # self.SL_tunnel_infos = jp.array(self.get_custom_tunnels_steeringlaw(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])))
        # self.SL_tunnel_infos = jp.array(self.get_custom_tunnels_different_lengths(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])) + 
        #                                     self.get_custom_tunnels_different_widths(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])))

        #TODO: hardcoded atm! should be data.site_xpos[self.screen_id][1:], but when this method is called, the data object has not yet been created
        screen_pos_center = jp.array([-0.27,  0.993], dtype=jp.float32)

        SL_tunnel_infos_list = self.get_custom_tunnels_steeringlaw(rng2, screen_pos_center=screen_pos_center, task_type=self.task_type, random_start_pos=self.random_start_pos)
        assert len(SL_tunnel_infos_list) > 0, f"For this task type ({self.task_type}), no tunnels have been returned."

        # check that all created instances share the same keys
        _key_list = [set(d.keys()) for d in SL_tunnel_infos_list]
        SL_tunnel_infos_keys = _key_list[0].intersection(*_key_list[1:])
        assert _key_list[0] == SL_tunnel_infos_keys, f"Not all created tunnel_info instances share the same keys! The following keys are missing from some instances: {_key_list[0] - SL_tunnel_infos_keys}"

        # merge list of dicts into one dict of jp.arrays (as these can be index with a traced int array), where first dimension corresponds to eval_id
        SL_tunnel_infos = {}
        for k in SL_tunnel_infos_keys:
            if not k in ("tunnel_angle_interp", "tunnel_extras"):
                SL_tunnel_infos[k] = jp.array(tuple(d[k] for d in SL_tunnel_infos_list))

        # store keys of tunnel_extras subdict separately, but unpack its values into the same level        
        _extra_key_list = [set(d["tunnel_extras"].keys()) if "tunnel_extras" in d.keys() else set() for d in SL_tunnel_infos_list]
        SL_tunnel_infos_keys_extra = _extra_key_list[0].union(*_extra_key_list[1:])
        for k in SL_tunnel_infos_keys_extra:
            SL_tunnel_infos[k] = jp.array(tuple(d["tunnel_extras"][k] if ("tunnel_extras" in d.keys()) and (k in d["tunnel_extras"].keys()) else jp.nan for d in SL_tunnel_infos_list))
        
        self.SL_tunnel_infos_keys = SL_tunnel_infos_keys - {"tunnel_extras", "tunnel_angle_interp"}
        self.SL_tunnel_infos = SL_tunnel_infos
        self.SL_tunnel_infos_keys_extra = SL_tunnel_infos_keys_extra
        self.n_randomizations = len(SL_tunnel_infos_list)
        return self.n_randomizations

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        # jax.debug.print('Step start - completed_phase_0: {}', state.info['completed_phase_0'])
        rng = state.info['rng']
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)
        
        # step forward
        ## TODO: can we move parts of this code into MyoUserBase.step (as a super method)?
        data = mjx_env.step(self.mjx_model, state.data, new_ctrl, n_substeps=self.n_substeps)
        # if self.vision or self.eval_mode:
        #     self.update_target_visuals(target_pos=state.info['target_pos'], target_radius=state.info['target_radius'])

        # # collect observations and reward
        if self.vision:
            # TODO: do we need to update target information for rendering?
            # data = self.add_target_pos_to_data(data, state.info["target_pos"])
            pixels_dict = self.generate_pixels(data, state.info['render_token'])
            state.info.update(pixels_dict)
        obs_dict = self.get_obs_dict(data, state.info)
        obs_dict = self.update_obs_with_pixels(obs_dict, state.info)
        obs = self.obsdict2obsvec(obs_dict)
        
        rwd_dict = self.get_reward_dict(obs_dict)
        #rwd_dict, state.info = self.get_reward_dict(obs_dict, state.info)
        _updated_info = self.update_info(state.info, obs_dict)
        _, _updated_info['rng'] = jax.random.split(rng, 2) #update rng after each step to ensure variability across steps
        state.replace(info=_updated_info)
        
        #done = rwd_dict['done']
        #jax.debug.print('self.terminate_out_of_bounds {}', self.terminate_out_of_bounds)
        done = self.terminate_out_of_bounds * jp.logical_or(rwd_dict['done'], rwd_dict["truncated"]).astype(jp.float32) + (1-self.terminate_out_of_bounds)*rwd_dict['done']
        
        # auxiliary variables used to aggregate certain metrics at the last step of an episode only
        # #TODO: allow for option in EpisodeWrapper to log final/mean/sum metrics
        # This is basically completed phase 0 or completed phase 1
        completed_phase_0_metric = obs_dict["completed_phase_0"] + obs_dict["completed_phase_1"] - (obs_dict["completed_phase_0"] * obs_dict["completed_phase_1"])

        crossed_cp_dict = {key: obs_dict[key] for key in obs_dict.keys() if key.startswith('crossed_checkpoint_')}
        state.metrics.update(
            completed_phase_0 = completed_phase_0_metric,
            completed_phase_1 = obs_dict["completed_phase_1"],
            dist = obs_dict["dist"],
            distance_phase_0 = obs_dict["distance_phase_0"],
            distance_phase_1 = obs_dict["distance_phase_1_metric"],
            phase_1_x_dist = obs_dict["phase_1_x_dist"],
            dist_to_start_line = obs_dict["dist_to_start_line"],
            dist_to_end_line = obs_dict["dist_to_end_line"],
            path_length = obs_dict["path_length"],
            #con_0_close_to_start_pos = obs_dict["con_0_close_to_start_pos"],
            #con_0_touching_screen = obs_dict["con_0_touching_screen"],
            #con_1_touching_screen = obs_dict["con_1_touching_screen"],
            #con_1_crossed_line_y = obs_dict["con_1_crossed_line_y"],
            percentage_achieved = obs_dict["path_percentage"],
            distance_to_tunnel_bounds = obs_dict["distance_to_tunnel_bounds"],
            softcons_for_bounds = obs_dict["softcons_for_bounds"],
            out_of_bounds = 1.-obs_dict["con_0_1_within_tunnel_limits"],
            not_touching = 1. * (1.0 - obs_dict["con_1_touching_screen"]) * obs_dict["completed_phase_0"],
            jac_effort_reward = rwd_dict["jac_effort"]*self.weighted_reward_keys['jac_effort'],
            #neural_effort_reward = rwd_dict["neural_effort"]*self.weighted_reward_keys['neural_effort'],
            distance_reward = rwd_dict['reach']*self.weighted_reward_keys['reach'],
            bonus_reward = rwd_dict['bonus_1']*self.weighted_reward_keys['bonus_1'],
            touch_reward = rwd_dict['phase_1_touch']*self.weighted_reward_keys['phase_1_touch'],
            #tunnel_reward = rwd_dict['phase_1_tunnel']*self.weighted_reward_keys['phase_1_tunnel'],
            tunnel_reward = self.weighted_reward_keys['phase_1_tunnel']*(1.*(1.-obs_dict['completed_phase_1'])*(obs_dict['completed_phase_0']*(-obs_dict['softcons_for_bounds']**self.weighted_reward_keys['power_for_softcons']) + (1.-obs_dict['completed_phase_0'])*(-1.))),
            **crossed_cp_dict,
        )

        # return self.forward(**kwargs)

        return state.replace(
            data=data, obs=obs, reward=rwd_dict['dense'], done=done
        )
    

    def update_task_visuals(self, mj_model, state):
        screen_pos = state.data.site_xpos[self.screen_id]
        nodes, _, _, _ = state.info["tunnel_nodes"], state.info["tunnel_boundary_left"], state.info["tunnel_boundary_right"], state.info["tunnel_boundary_parametrization"] 
        nodes_left, nodes_right = state.info["tunnel_nodes_left"], state.info["tunnel_nodes_right"]

        n_connectors = len(nodes) - 1
        rel_vec = np.array([-1, 0])  #use [-1, 0] as reference vector for horizontal axis (i.e. x-axis in relevative plane coordinates), as global y-axis used as x-coordinate points to left in MuJoCo

        mid_points_left = 0.5*(nodes_left[:-1] + nodes_left[1:])
        connector_vecs_left = nodes_left[1:] - nodes_left[:-1]
        segment_lengths_left = np.linalg.norm(connector_vecs_left, axis=1)
        segments_angles_left = np.array([(np.arctan2(-(cross2d(_vec, rel_vec)), np.dot(_vec, rel_vec)) + np.pi) % (2*np.pi) - np.pi for _vec in connector_vecs_left])

        mid_points_right = 0.5*(nodes_right[:-1] + nodes_right[1:])
        connector_vecs_right = nodes_right[1:] - nodes_right[:-1]
        segment_lengths_right = np.linalg.norm(connector_vecs_right, axis=1)
        segments_angles_right = np.array([(np.arctan2(-(cross2d(_vec, rel_vec)), np.dot(_vec, rel_vec)) + np.pi) % (2*np.pi) - np.pi for _vec in connector_vecs_right])

        for i in range(n_connectors):
            # left segments
            mj_model.site(f"line_{i}").pos[1:] = mid_points_left[i] - screen_pos[1:]
            mj_model.site(f"line_{i}").size[1] = 0.5*segment_lengths_left[i]  # Use y-dimension for length
            mj_model.site(f"line_{i}").size[2] = 0.005  # Keep fixed width in z-dimension
            # Convert angle to quaternion rotation around x-axis
            quat = Rotation.from_euler("x", segments_angles_left[i]).as_quat(canonical=True)
            mj_model.site(f"line_{i}").quat[:] = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to scalar-first format
            mj_model.site(f"line_{i}").rgba[:] = np.array([1., 0., 0., 0.8])

            # right segments
            mj_model.site(f"line_{n_connectors+i}").pos[1:] = mid_points_right[i] - screen_pos[1:]
            mj_model.site(f"line_{n_connectors+i}").size[1] = 0.5*segment_lengths_right[i]  # Use y-dimension for length
            mj_model.site(f"line_{n_connectors+i}").size[2] = 0.005  # Keep fixed width in z-dimension
            # Convert angle to quaternion rotation around x-axis
            quat = Rotation.from_euler("x", segments_angles_right[i]).as_quat(canonical=True)
            mj_model.site(f"line_{n_connectors+i}").quat[:] = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to scalar-first format
            mj_model.site(f"line_{n_connectors+i}").rgba[:] = np.array([0., 1., 0., 0.8])

        # if self.task_type == "menu_0":
        #     n_connectors = len(nodes) - 1
        #     rel_vec = np.array([-1, 0])  #use [-1, 0] as reference vector for horizontal axis (i.e. x-axis in relevative plane coordinates), as global y-axis used as x-coordinate points to left in MuJoCo

        #     mid_points_left = 0.5*(nodes_left[:-1] + nodes_left[1:])
        #     connector_vecs_left = nodes_left[1:] - nodes_left[:-1]
        #     segment_lengths_left = np.linalg.norm(connector_vecs_left, axis=1)
        #     segments_angles_left = np.array([(np.arctan2(-(cross2d(_vec, rel_vec)), np.dot(_vec, rel_vec)) + np.pi) % (2*np.pi) - np.pi for _vec in connector_vecs_left])

        #     mid_points_right = 0.5*(nodes_right[:-1] + nodes_right[1:])
        #     connector_vecs_right = nodes_right[1:] - nodes_right[:-1]
        #     segment_lengths_right = np.linalg.norm(connector_vecs_right, axis=1)
        #     segments_angles_right = np.array([(np.arctan2(-(cross2d(_vec, rel_vec)), np.dot(_vec, rel_vec)) + np.pi) % (2*np.pi) - np.pi for _vec in connector_vecs_right])

        #     # TODO: this workaround only works for perfectly vertical and/or horizontal boundaries; in other cases, a fix is needed that allows to change site orientation on the fly...  
        #     # For example, we could wrap each site into a body and apply all transformations to the body rather than to the site
        #     # Also, how to render arbitrarily curved paths?
        #     height_width_indices_left = 1 + (np.abs(segments_angles_left // (np.pi/2)).astype(int))
        #     height_width_indices_right = 1 + (np.abs(segments_angles_right // (np.pi/2)).astype(int))

        #     for i in range(n_connectors):
        #         # left segments
        #         mj_model.site(f"line_{i}").pos[1:] = mid_points_left[i] - screen_pos[1:]
        #         mj_model.site(f"line_{i}").size[height_width_indices_left[i]] = 0.5*segment_lengths_left[i]
        #         # mj_model.site(f"line_{i}").quat[:] = scipy.spatial.transform.Rotation.from_euler("x", segments_angles_left[i]).as_quat(scalar_first=True)
        #         mj_model.site(f"line_{i}").rgba[:] = np.array([1., 0., 0., 0.8])

        #         # right segments
        #         mj_model.site(f"line_{n_connectors+i}").pos[1:] = mid_points_right[i] - screen_pos[1:]
        #         mj_model.site(f"line_{n_connectors+i}").size[height_width_indices_right[i]] = 0.5*segment_lengths_right[i]
        #         # mj_model.site(f"line_{n_connectors+i}").quat[:] = scipy.spatial.transform.Rotation.from_euler("x", segments_angles_right[i]).as_quat(scalar_first=True)
        #         mj_model.site(f"line_{n_connectors+i}").rgba[:] = np.array([1., 0., 0., 0.8])
        # elif self.task_type == "circle_0":
        #     tunnel_center, circle_radius, tunnel_size = state.info["tunnel_extras"]["tunnel_center"], state.info["tunnel_extras"]["circle_radius"], state.info["tunnel_extras"]["tunnel_size"]
        #     inner_radius, outer_radius = circle_radius - 0.5* tunnel_size, circle_radius + 0.5* tunnel_size

        #     # outer circle boundary
        #     mj_model.site("circle_0").pos[1:] = tunnel_center - screen_pos[1:]
        #     mj_model.site("circle_0").size[0] = outer_radius
        #     mj_model.site("circle_0").rgba[:] = np.array([0., 1., 0., 0.8])
            
        #     # inner circle boundary
        #     mj_model.site("circle_1").pos[1:] = tunnel_center - screen_pos[1:]
        #     mj_model.site("circle_1").size[0] = inner_radius
        #     mj_model.site("circle_1").rgba[:] = np.array([1., 0., 0., 0.8])

        # start pos
        mj_model.site("refpos_0").pos[1:] = nodes[0] - screen_pos[1:]
        mj_model.site("refpos_0").rgba[:] = np.array([0., 1., 0., 0.8])

        # end pos
        mj_model.site("refpos_1").pos[1:] = nodes[-1] - screen_pos[1:]
        mj_model.site("refpos_1").rgba[:] = np.array([0., 0., 1., 0.8])

    def eval_metrics(self, movement_times, rollout_states, task_type=None, average_r2=True, plot_data=False):
        if task_type is None:
            task_type = self._config.task_config.type

        eval_metrics = calculate_steering_laws(movement_times=movement_times, rollout_states=rollout_states, task=task_type, average_r2=average_r2, plot_data=plot_data)

        return eval_metrics