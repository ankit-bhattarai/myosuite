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

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
from myosuite.envs.myo.fatigue import CumulativeFatigue
from myosuite.envs.myo.myouser.base import MyoUserBase, BaseEnvConfig
from dataclasses import dataclass, field
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

@dataclass
class SteeringTaskConfig:
    distance_reach_metric_coefficient: float = 10.
    screen_distance_x: float = 0.5
    screen_friction: float = 0.1
    ee_name: str = "fingertip"
    obs_keys: List[str] = field(default_factory=lambda: ['qpos', 'qvel', 'qacc', 'fingertip', 'act'])
    omni_keys: List[str] = field(default_factory=lambda: ['screen_pos', 'start_line', 'end_line', 'top_line', 'bottom_line', 'completed_phase_0_arr', 'target'])
    weighted_reward_keys: Dict[str, float] = field(default_factory=lambda: {
        "reach": 1,
        "bonus_1": 10,
        "phase_1_touch": 1,
        "phase_1_tunnel": 3,
        "neural_effort": 0,
        "jac_effort": 0,
        "power_for_softcons": 15,
        "truncated": -10,
        "truncated_progress": -20
    })
    max_duration: float = 4.
    max_trials: int = 1
    reset_type: str = "epsilon_uniform"
    min_width: float = 0.03
    max_width: float = 0.6
    min_height: float = 0.03
    max_height: float = 0.1
    bottom: float = -0.3
    top: float = 0.3
    left: float = 0.3
    right: float = -0.2
    terminate_out_of_bounds: float = 1.0
    min_dwell_phase_0: float = 0.
    min_dwell_phase_1: float = 0.

@dataclass
class SteeringEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserSteering"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_steering_myouser.xml"
    task_config: SteeringTaskConfig = field(default_factory=lambda: SteeringTaskConfig())

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
            omni_keys=['screen_pos', 'start_line', 'end_line', 'top_line', 'bottom_line', 'completed_phase_0_arr', 'target'],
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


class MyoUserSteering(MyoUserBase): 
    def modify_mj_model(self, mj_model):
        default_screen_distance_x = 0.50  #distance from humphant body; corresponds to default screen pos[0] value in xml file

        mj_model.body('screen').pos[0] += self._config.task_config.screen_distance_x - default_screen_distance_x
        mj_model.geom('screen').friction = self._config.task_config.screen_friction
        if any([mj_model.geom(i).name=='fingertip_contact' for i in range(mj_model.ngeom)]):
            mj_model.geom('fingertip_contact').friction = self._config.task_config.screen_friction
        return mj_model
        
    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if (geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER) or (geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec    
   
    def _setup(self):
        """Task specific setup"""
        super()._setup()
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
        self.top_line_id = self._mj_model.site("top_line").id
        self.bottom_line_id = self._mj_model.site("bottom_line").id
        self.start_line_id = self._mj_model.site("start_line").id
        self.end_line_id = self._mj_model.site("end_line").id

        self.ee_name = self._config.task_config.ee_name
        self.fingertip_id = self._mj_model.site(self.ee_name).id

        #self._shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_rot")
        #self._elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flexion")
        # self.screen_touch_id = self._mj_model.sensor("screen_touch").id

        #TODO: once contact sensors are integrated, check if the fingertip_geom is needed or not

        self.distance_reach_metric_coefficient = self._config.task_config.distance_reach_metric_coefficient

        # Currently hardcoded
        self.min_width = self._config.task_config.min_width
        self.max_width = self._config.task_config.max_width
        self.min_height = self._config.task_config.min_height
        self.max_height = self._config.task_config.max_height
        self.bottom = self._config.task_config.bottom
        self.top = self._config.task_config.top
        self.left = self._config.task_config.left
        self.right = self._config.task_config.right
        self.terminate_out_of_bounds = self._config.task_config.terminate_out_of_bounds
        self.min_dwell_phase_0 = self._config.task_config.min_dwell_phase_0
        self.phase_0_completed_min_steps = max(np.ceil(self._config.task_config.min_dwell_phase_0 / self._config.ctrl_dt).astype(int), 1)
        self.min_dwell_phase_1 = self._config.task_config.min_dwell_phase_1
        self.phase_1_completed_min_steps = max(np.ceil(self._config.task_config.min_dwell_phase_1 / self._config.ctrl_dt).astype(int), 1)

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
        
        # Store current control input
        obs_dict['last_ctrl'] = data.ctrl.copy()

        # End-effector and target position - read current position from data instead of cached info
        obs_dict['fingertip'] = data.site_xpos[self.fingertip_id]
        obs_dict['screen_pos'] = data.site_xpos[self.screen_id]

        # # Read positions directly from current data instead of stale info
        obs_dict['start_line'] = info['start_line']
        obs_dict['end_line'] = info['end_line']
        obs_dict['top_line'] = info['top_line']
        obs_dict['bottom_line'] = info['bottom_line']
        # obs_dict['touching_screen'] = data.sensordata[self.screen_touch_id] > 0.0

        phase_0_completed_steps = info['phase_0_completed_steps']
        phase_1_completed_steps = info['phase_1_completed_steps']
        completed_phase_0 = info['completed_phase_0']
        completed_phase_1 = info['completed_phase_1']
        ee_pos = obs_dict['fingertip']
        start_line = obs_dict['start_line']
        end_line = obs_dict['end_line']
        bottom_line_z = obs_dict['bottom_line'][2]
        top_line_z = obs_dict['top_line'][2]
        path_length = jp.linalg.norm(end_line[1] - start_line[1])
        path_width = jp.linalg.norm(top_line_z - bottom_line_z)
        dist_to_start_line = jp.linalg.norm(ee_pos - start_line, axis=-1)
        dist_to_end_line = jp.linalg.norm(ee_pos[1] - end_line[1])

        obs_dict['percentage_of_remaining_path'] = dist_to_end_line/path_length

        # Update phase immediately based on current position
        touching_screen_phase_0 = 1.0 *(jp.linalg.norm(ee_pos[0] - obs_dict['screen_pos'][0]) <= 0.01)
        within_z_limits = 1.0 * (ee_pos[2] >= bottom_line_z) * (ee_pos[2] <= top_line_z)
        within_y_dist = 1.0 * (jp.linalg.norm(ee_pos[1] - start_line[1]) <= 0.01)
        phase_0_completed_now = touching_screen_phase_0 * within_z_limits * within_y_dist
        phase_0_completed_steps = (phase_0_completed_steps + 1) * phase_0_completed_now
        completed_phase_0 = completed_phase_0 + (1 - completed_phase_0) * (phase_0_completed_steps >= self.phase_0_completed_min_steps)
        
        crossed_line_y = 1.0 * (ee_pos[1] <= end_line[1])
        phase_1_x_dist = jp.linalg.norm(ee_pos[0] - obs_dict['screen_pos'][0])
        touching_screen_phase_1 = 1.0 * (phase_1_x_dist <= 0.01)
        phase_1_completed_now = completed_phase_0 * crossed_line_y * touching_screen_phase_1 * within_z_limits
        phase_1_completed_steps = (phase_1_completed_steps + 1) * phase_1_completed_now
        completed_phase_1 = completed_phase_1 + (1 - completed_phase_1) * (phase_1_completed_steps >= self.phase_1_completed_min_steps)   

        relative_position = jp.linalg.norm(ee_pos[2] - bottom_line_z)
        softcons_for_bounds = jp.clip(jp.abs(relative_position) / (path_width / 2), 0, 1)
        
        # Reset phase 0 when phase 1 is done (and episode ends)
        ## TODO: delay this update to the step function, to ensure consistency between different observations (e.g. when defining the reward function)?
        completed_phase_0 = completed_phase_0 * (1. - completed_phase_1)

        obs_dict["con_0_touching_screen"] = touching_screen_phase_0
        obs_dict["con_0_1_within_z_limits"] = within_z_limits
        obs_dict["con_0_within_y_dist"] = within_y_dist
        obs_dict["completed_phase_0"] = completed_phase_0
        obs_dict['completed_phase_0_arr'] = jp.array([completed_phase_0])
        obs_dict["con_1_crossed_line_y"] = crossed_line_y
        obs_dict["con_1_touching_screen"] = touching_screen_phase_1
        obs_dict["completed_phase_1"] = completed_phase_1
        obs_dict["softcons_for_bounds"] = softcons_for_bounds

        ## Compute distances

        phase_0_distance = dist_to_start_line + path_length
        phase_1_distance = dist_to_end_line
        dist = completed_phase_0 * phase_1_distance + (1. - completed_phase_0) * phase_0_distance
        
        obs_dict["distance_phase_0"] = (1. - completed_phase_0) * phase_0_distance
        obs_dict["distance_phase_1"] = completed_phase_0 * phase_1_distance
        obs_dict["dist"] = dist
        obs_dict["phase_1_x_dist"] = phase_1_x_dist

        ## Additional observations
        obs_dict['target'] = completed_phase_0 * obs_dict['end_line'] + (1. - completed_phase_0) * obs_dict['start_line']
        # obs_dict["completed_phase_0_first"] = (1. - info["completed_phase_0"]) * (obs_dict["completed_phase_0"])
        # obs_dict["completed_phase_1_first"] = (1. - info["completed_phase_1"]) * (obs_dict["completed_phase_1"])
        obs_dict["phase_0_completed_steps"] = phase_0_completed_steps
        obs_dict["phase_1_completed_steps"] = phase_1_completed_steps

        return obs_dict
    
    def update_info(self, info, obs_dict):
        # TODO: is this really needed? can we drop (almost all) info keys?
        info['last_ctrl'] = obs_dict['last_ctrl']
        # info['motor_act'] = obs_dict['motor_act']
        info['fingertip'] = obs_dict['fingertip']
        info["phase_0_completed_steps"] = obs_dict["phase_0_completed_steps"]
        info["phase_1_completed_steps"] = obs_dict["phase_1_completed_steps"]
        info['completed_phase_0'] = obs_dict['completed_phase_0']
        info['completed_phase_1'] = obs_dict['completed_phase_1']

        return info
    
    def get_reward_dict(self, obs_dict):#, info):

        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)

        # Give some intermediate reward for transitioning from phase 0 to phase 1 but only when finger is touching the
        # start line when in phase 0   

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            # ('reach',   1.*(jp.exp(-obs_dict["dist"]*self.distance_reach_metric_coefficient) - 1.)/self.distance_reach_metric_coefficient),  #-1.*reach_dist)
            ('reach',   -1.*(1.-(obs_dict['phase_1_completed_steps']>0))*obs_dict["dist"]),  #-1.*reach_dist)
            #('bonus_0',   1.*(1.-obs_dict['completed_phase_1'])*((1.-obs_dict['completed_phase_0'])*(obs_dict['con_0_touching_screen']))),  #TODO: possible alternative: give one-time bonus when obs_dict['completed_phase_0_first']==True
            ('bonus_1',   1.*(obs_dict['completed_phase_1'])),  #TODO :use obs_dict['completed_phase_1_first'] instead?
            ('phase_1_touch',   1.*(obs_dict['completed_phase_0']*(-obs_dict['phase_1_x_dist']) + (1.-obs_dict['completed_phase_0'])*(-0.3))),
            #('phase_1_touch',   -1.*(obs_dict['completed_phase_0']*(1-obs_dict['con_1_touching_screen']) + (1.-obs_dict['completed_phase_0'])*(0.5))),
            #('phase_1_tunnel', 1.*(1.-obs_dict['completed_phase_1'])*(obs_dict['completed_phase_0']*(-obs_dict['softcons_for_bounds']**15) + (1.-obs_dict['completed_phase_0'])*(-1.))),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            ('jac_effort', -1.* self.get_jac_effort_costs(obs_dict)),
            ('truncated', 1.*(1.-obs_dict["con_0_1_within_z_limits"])*obs_dict["completed_phase_0"])),#jp.logical_or(,(1.0 - obs_dict["con_1_touching_screen"]) * obs_dict["completed_phase_0"])
            ('truncated_progress', 1.*((1.-obs_dict["con_0_1_within_z_limits"])*obs_dict['completed_phase_0']*obs_dict['percentage_of_remaining_path']),
            # # Must keys
            ('done',    1.*(obs_dict['completed_phase_1'])), #np.any(reach_dist > far_th))),
        ))

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
    
    def get_ejk_effort_costs(self, obs_dict, info):
        r_effort = jp.mean(obs_dict['last_ctrl'])

        qacc = obs_dict['qacc']
        r_jerk = (jp.norm(qacc - info['previous_qacc']) / self.sim_dt) / 100000.0

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
            'top_line': data.site_xpos[self.top_line_id],
            'bottom_line': data.site_xpos[self.bottom_line_id],
            'start_line': data.site_xpos[self.start_line_id],
            'end_line': data.site_xpos[self.end_line_id],
        }
    
    def get_custom_tunnel_centers(self, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        rng1, rng2 = jax.random.split(rng, 2)
        bottom_line, top_line = self.get_tunnel_limits(rng1, self.bottom, self.top, self.min_height, self.max_height)
        right_line, left_line = self.get_tunnel_limits(rng2, self.right, self.left, self.min_width, self.max_width)
        return bottom_line, top_line, left_line, right_line

    def get_custom_tunnel(self, rng: jax.Array, data: mjx.Data) -> dict[str, jax.Array]:
        bottom_line, top_line, left_line, right_line = self.get_custom_tunnel_centers(rng)
        width_midway = (left_line + right_line) / 2
        height_midway = (top_line + bottom_line) / 2
        relevant_positions = self.get_relevant_positions(data)
        tunnel_positions = {}
        tunnel_positions['bottom_line'] = relevant_positions['screen_pos'] + jp.array([0., width_midway, bottom_line])
        tunnel_positions['top_line'] = relevant_positions['screen_pos'] + jp.array([0., width_midway, top_line])
        tunnel_positions['start_line'] = relevant_positions['screen_pos'] + jp.array([0., left_line, height_midway])
        tunnel_positions['end_line'] = relevant_positions['screen_pos'] + jp.array([0., right_line, height_midway])
        tunnel_positions['screen_pos'] = relevant_positions['screen_pos']
        #jax.debug.print("width_midway: {}, height_midway: {}", width_midway, height_midway)
        return tunnel_positions
    
    def random_positions_steeringlaw(self, rng: jax.Array, L, W, right):
        pos_min, pos_max = -0.2, 0.3
        left = right - L
        rng, rng2 = jax.random.split(rng, 2)
        bottom = jax.random.uniform(rng, minval=pos_min, maxval=pos_max-W)
        top = bottom + W
        return left, bottom, top
    
    def get_custom_tunnels_different_lengths(self, rng: jax.Array, screen_pos: jax.Array) -> dict[str, jax.Array]:
        tunnel_positions_different_lengths  = []
        IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        right = 0.3
        top = 0.025
        bottom = -0.025
        W = top - bottom
        for ID in IDs:
            combos = 0
            while combos < 1:
                L = ID * W
                tunnel_positions = []
                left = right - L
                width_midway = (left + right) / 2
                height_midway = (top + bottom) / 2
                tunnel_positions.append(screen_pos + jp.array([0., width_midway, bottom]))
                tunnel_positions.append(screen_pos + jp.array([0., width_midway, top]))
                tunnel_positions.append(screen_pos + jp.array([0., right, height_midway]))
                tunnel_positions.append(screen_pos + jp.array([0., left, height_midway]))
                tunnel_positions.append(screen_pos)
                combos += 1

                # for i in range(20):
                tunnel_positions_different_lengths.append(tunnel_positions)
        return tunnel_positions_different_lengths
    
    def get_custom_tunnels_different_widths(self, rng: jax.Array, screen_pos: jax.Array) -> dict[str, jax.Array]:
        tunnel_positions_different_widths  = []
        IDs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        right = 0.3
        left = -0.2
        top = 0.1
        L = right - left

        for ID in IDs:
            combos = 0
            while combos < 1:
                W = L/ID
                bottom = top - W
                tunnel_positions = []
                left = right - L
                width_midway = (left + right) / 2
                height_midway = (top + bottom) / 2
                tunnel_positions.append(screen_pos + jp.array([0., width_midway, bottom]))
                tunnel_positions.append(screen_pos + jp.array([0., width_midway, top]))
                tunnel_positions.append(screen_pos + jp.array([0., right, height_midway]))
                tunnel_positions.append(screen_pos + jp.array([0., left, height_midway]))
                tunnel_positions.append(screen_pos)
                combos += 1

                # for i in range(20):
                tunnel_positions_different_widths.append(tunnel_positions)
        return tunnel_positions_different_widths

    # def get_custom_tunnels_steeringlaw(self, rng: jax.Array, screen_pos: jax.Array) -> dict[str, jax.Array]:
    #     tunnel_positions_total  = []
    #     tunnel_positions_total = self.get_custom_tunnels_different_lengths(rng, screen_pos)
    #     #tunnel_positions_total.append(self.get_custom_tunnels_different_widths(rng, screen_pos))
    #     return tunnel_positions_total

    def get_custom_tunnels_steeringlaw(self, rng: jax.Array, screen_pos: jax.Array) -> dict[str, jax.Array]:
        tunnel_positions_total  = []
        IDs = [1, 2, 3, 4, 5]
        L_min, L_max = 0.05, 0.5
        W_min, W_max = 0.07, 0.6
        right = 0.3
        for ID in IDs:
            combos = 0
            while combos < 1:
                rng, rng2 = jax.random.split(rng, 2)
                W = jax.random.uniform(rng, minval=W_min, maxval=W_max)
                L = ID * W
                if L_min <= L <= L_max:
                    tunnel_positions = []  #{}
                    left, bottom, top = self.random_positions_steeringlaw(rng2, L, W, right)
                    width_midway = (left + right) / 2
                    height_midway = (top + bottom) / 2
                    tunnel_positions.append(screen_pos + jp.array([0., width_midway, bottom]))
                    tunnel_positions.append(screen_pos + jp.array([0., width_midway, top]))
                    tunnel_positions.append(screen_pos + jp.array([0., right, height_midway]))
                    tunnel_positions.append(screen_pos + jp.array([0., left, height_midway]))
                    tunnel_positions.append(screen_pos)
                    combos += 1

                    for i in range(20):
                        tunnel_positions_total.append(tunnel_positions)
        return tunnel_positions_total

    def reset(self, rng, render_token=None, tunnel_positions=None):
        # jax.debug.print(f"RESET INIT")

        _, rng = jax.random.split(rng, 2)

        # Reset biomechanical model
        data = self._reset_bm_model(rng)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)

        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                "phase_0_completed_steps": 0,
                "phase_1_completed_steps": 0,
                "completed_phase_0": 0.0,
                "completed_phase_1": 0.0,
                "previous_qacc": 0.0,
                }
        if tunnel_positions is None:
            info.update(self.get_custom_tunnel(rng, data))
        else:
            info.update(tunnel_positions)

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
            #'con_0_touching_screen': 0.0,
            #'con_1_touching_screen': 0.0,
            #'con_1_crossed_line_y': 0.0,
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
        
        return State(data, obs, reward, done, metrics, info)
    
    def auto_reset(self, rng, info_before_reset, **kwargs):
        render_token = info_before_reset["render_token"] if self.vision else None
        return self.reset(rng, render_token=render_token, **kwargs)
    
    def eval_reset(self, rng, eval_id, **kwargs):
        """Reset function wrapper called by evaluate_policy."""
        _tunnel_position_jp = self.SL_tunnel_positions.at[eval_id].get()
        tunnel_positions = {}
        tunnel_positions['bottom_line'] = _tunnel_position_jp[0]
        tunnel_positions['top_line'] = _tunnel_position_jp[1]
        tunnel_positions['start_line'] = _tunnel_position_jp[2]
        tunnel_positions['end_line'] = _tunnel_position_jp[3]
        tunnel_positions['screen_pos'] = _tunnel_position_jp[4]

        return self.reset(rng, tunnel_positions=tunnel_positions, **kwargs)
    
    def prepare_eval_rollout(self, rng, **kwargs):
        """Function that can be used to define random parameters to be used across multiple evaluation rollouts/resets.
        May return the number of evaluation episodes that should be rolled out (before this method should be called again)."""
        
        ## Setup evaluation episodes for Steering Law validation
        rng, rng2 = jax.random.split(rng, 2)
        # self.SL_tunnel_positions = jp.array(self.get_custom_tunnels_different_lengths(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])))
        # self.SL_tunnel_positions = jp.array(self.get_custom_tunnels_steeringlaw(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])))
        self.SL_tunnel_positions = jp.array(self.get_custom_tunnels_different_lengths(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])) + 
                                            self.get_custom_tunnels_different_widths(rng2, screen_pos=jp.array([0.532445, -0.27, 0.993])))

        return len(self.SL_tunnel_positions)

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

        state.metrics.update(
            completed_phase_0 = obs_dict["completed_phase_0"],
            completed_phase_1 = obs_dict["completed_phase_1"],
            dist = obs_dict["dist"],
            distance_phase_0 = obs_dict["distance_phase_0"],
            distance_phase_1 = obs_dict["distance_phase_1"],
            phase_1_x_dist = obs_dict["phase_1_x_dist"],
            #con_0_touching_screen = obs_dict["con_0_touching_screen"],
            #con_1_touching_screen = obs_dict["con_1_touching_screen"],
            #con_1_crossed_line_y = obs_dict["con_1_crossed_line_y"],
            softcons_for_bounds = obs_dict["softcons_for_bounds"],
            out_of_bounds = 1.-obs_dict["con_0_1_within_z_limits"],
            not_touching = 1. * (1.0 - obs_dict["con_1_touching_screen"]) * obs_dict["completed_phase_0"],
            jac_effort_reward = rwd_dict["jac_effort"]*self.weighted_reward_keys['jac_effort'],
            #neural_effort_reward = rwd_dict["neural_effort"]*self.weighted_reward_keys['neural_effort'],
            distance_reward = rwd_dict['reach']*self.weighted_reward_keys['reach'],
            bonus_reward = rwd_dict['bonus_1']*self.weighted_reward_keys['bonus_1'],
            touch_reward = rwd_dict['phase_1_touch']*self.weighted_reward_keys['phase_1_touch'],
            #tunnel_reward = rwd_dict['phase_1_tunnel']*self.weighted_reward_keys['phase_1_tunnel'],
            tunnel_reward = self.weighted_reward_keys['phase_1_tunnel']*(1.*(1.-obs_dict['completed_phase_1'])*(obs_dict['completed_phase_0']*(-obs_dict['softcons_for_bounds']**self.weighted_reward_keys['power_for_softcons']) + (1.-obs_dict['completed_phase_0'])*(-1.)))
        )

        # return self.forward(**kwargs)

        return state.replace(
            data=data, obs=obs, reward=rwd_dict['dense'], done=done
        )
    

    def update_task_visuals(self, mj_model, state):
        screen_pos = state.info["screen_pos"] + jp.array([0.01, 0., 0.])  #need to re-introduce site pos offset from xml file that was ignored in get_custom_tunnel() to ensure that task visuals properly appear in front of the screen 
        screen_y = screen_pos[1]
        screen_z = screen_pos[2]
        top_line = state.info["top_line"]
        bottom_line = state.info["bottom_line"]
        start_line = state.info["start_line"]
        end_line = state.info["end_line"]
        bottom_z = bottom_line[2] - screen_z
        top_z = top_line[2] - screen_z
        left_y = start_line[1] - screen_y
        right_y = end_line[1] - screen_y
        width_midway = (left_y + right_y) / 2
        height_midway = (top_z + bottom_z) / 2
        height = top_z - bottom_z
        width = left_y - right_y
        
        mj_model.site('bottom_line').pos[1:] = jp.array([width_midway, bottom_z])
        mj_model.site('bottom_line').size[1] = width / 2

        mj_model.site('top_line').pos[1:] = jp.array([width_midway, top_z])
        mj_model.site('top_line').size[1] = width / 2

        mj_model.site('start_line').pos[1:] = jp.array([left_y, height_midway])
        mj_model.site('start_line').size[2] = height / 2

        mj_model.site('end_line').pos[1:] = jp.array([right_y, height_midway])
        mj_model.site('end_line').size[2] = height / 2


    def calculate_metrics(self, rollout, eval_metrics_keys={"R^2"}):

        eval_metrics = {}

        # TODO: set eval_metrics_keys as config param?
        if True:  #"R^2" in eval_metrics_keys:
            a,b,r2,_ = self.calculate_r2(rollout)
            eval_metrics["eval/R^2"] = r2
            eval_metrics["eval/a"] = a
            eval_metrics["eval/b"] = b

        return eval_metrics

    def calculate_r2(self, rollouts, average_r2=True):
        MTs = jp.array([(rollout[np.argwhere(_compl_1)[0].item()].data.time - rollout[np.argwhere(_compl_0)[0].item()].data.time) 
                        for rollout in rollouts if any(_compl_0 := [r.metrics["completed_phase_0"] for r in rollout]) and 
                                                any(_compl_1 := [r.metrics["completed_phase_1"] for r in rollout])])
        Ds = jp.array([jp.abs(rollout[np.argwhere(_compl_0)[0].item()].info["end_line"][1] - rollout[np.argwhere(_compl_0)[0].item()].info["start_line"][1])
                        for rollout in rollouts if any(_compl_0 := [r.metrics["completed_phase_0"] for r in rollout]) and 
                                                any(_compl_1 := [r.metrics["completed_phase_1"] for r in rollout])])
        Ws = jp.array([jp.abs(rollout[np.argwhere(_compl_0)[0].item()].info["top_line"][2] - rollout[np.argwhere(_compl_0)[0].item()].info["bottom_line"][2])
                        for rollout in rollouts if any(_compl_0 := [r.metrics["completed_phase_0"] for r in rollout]) and 
                                                any(_compl_1 := [r.metrics["completed_phase_1"] for r in rollout])])
        IDs = (Ds / Ws).reshape(-1, 1)

        if len(IDs) == 0 or len(MTs) == 0:
            return np.nan, np.nan, np.nan, {}

        if average_r2:
            # Fit linear curve to mean per ID and compute R^2
            IDs_rounded = IDs.round(2)
            ID_means = jp.sort(jp.unique(IDs_rounded)).reshape(-1, 1)
            MT_means = jp.array([MTs[np.argwhere(IDs_rounded.flatten() == _id)].mean() for _id in ID_means])

            model = LinearRegression()
            model.fit(ID_means, MT_means)
            a = model.intercept_
            b = model.coef_[0]
            y_pred = model.predict(ID_means)
            r2 = r2_score(MT_means, y_pred)
            #print(f"IDs: {IDs}")
            #print(f"MTs: {MTs}")
        else:
            # Fit linear curve to all data points and compute R^2
            model = LinearRegression()
            model.fit(IDs, MTs)
            a = model.intercept_
            b = model.coef_[0]
            y_pred = model.predict(IDs)
            r2 = r2_score(MTs, y_pred)
            #print(f"IDs: {IDs}")
            #print(f"MTs: {MTs}")

        print(f"R^2: {r2}, a,b: {a},{b}")

        sl_data = {"ID": IDs, "MT_ref": MTs,
                "MT_pred": y_pred,
                "D": Ds, "W": Ws}
        if average_r2:
            sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means})

        return a,b,r2,sl_data