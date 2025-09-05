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
from dataclasses import dataclass, field
from typing import List, Dict

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
from myosuite.envs.myo.fatigue import CumulativeFatigue
from myosuite.envs.myo.myouser.base import MyoUserBase, BaseEnvConfig
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#Current state: Circle center is always screen position!! Otherwise: change it!!

@dataclass
class CircularSteeringTaskConfig:
    screen_distance_x: float = 0.5
    screen_friction: float = 0.1
    obs_keys: List[str] = field(default_factory=lambda: ['qpos', 'qvel', 'qacc', 'fingertip', 'act'])
    omni_keys: List[str] = field(default_factory=lambda: ['screen_pos', 'start_line', 'end_line', 'top_line_radius', 'bottom_line_radius', 'completed_phase_0', 'target', 'middle_line_crossed'])
    weighted_reward_keys: Dict[str, float] = field(default_factory=lambda: {
        "reach": 1,
        "bonus_1": 10,
        "phase_1_touch": 8,
        "phase_1_tunnel": 0,
        "neural_effort": 0,
        "jac_effort": 0,
        "power_for_softcons": 15,
        "truncated": 0,
        "truncated_progress": 0,
        "bonus_inside_path": 0
    })
    max_duration: float = 5.
    max_trials: int = 1
    reset_type: str = "epsilon_uniform"
    min_width: float = 0.1
    max_width: float = 0.2
    min_inner_radius: float = 0.05
    max_inner_radius: float = 0.3
    inner_radius: float = 0.1
    outer_radius: float = 0.2

@dataclass
class CircularSteeringEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserCircularSteering"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_circular_steering_myouser.xml"
    task_config: CircularSteeringTaskConfig = field(default_factory=lambda: CircularSteeringTaskConfig())

def default_config() -> config_dict.ConfigDict:
    #TODO: update/make use of env_config parameters!
    env_config = config_dict.create(
        model_path="myosuite/envs/myo/assets/arm/mobl_arms_index_circular_steering_myouser.xml",
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
            screen_distance_x=0.5,  #0.59
            screen_friction=0.1,
            obs_keys=['qpos', 'qvel', 'qacc', 'fingertip', 'act'], 
            omni_keys=['screen_pos', 'start_line', 'end_line', 'top_line', 'bottom_line', 'completed_phase_0_arr', 'target'],
             weighted_reward_keys=config_dict.create(
                reach=1,
                bonus_1=50,
                phase_1_touch=8,
                phase_1_tunnel=0,  #-2,
                neural_effort=0.0,  #1e-4,
                jac_effort=0,
                truncated=0, #0
                truncated_progress=-10,
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
        num_envs=1,  #8192,
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

class MyoUserCircularSteering(MyoUserBase): 
    def modify_mj_model(self, mj_model):
        mj_model.body('screen').pos[:] = jp.array([self._config.task_config.screen_distance_x, mj_model.body('screen').pos[1], mj_model.body('screen').pos[2]])
        mj_model.geom('screen').friction = self._config.task_config.screen_friction
        mj_model.geom('fingertip_contact').friction = self._config.task_config.screen_friction
        return mj_model
   
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
        self.fingertip_id = self._mj_model.site("fingertip").id
        # self.screen_touch_id = self._mj_model.sensor("screen_touch").id

        #TODO: once contact sensors are integrated, check if the fingertip_geom is needed or not

        self.min_width = self._config.task_config.min_width
        self.max_width = self._config.task_config.max_width
        self.min_inner_radius = self._config.task_config.min_inner_radius
        self.max_inner_radius = self._config.task_config.max_inner_radius
        
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
        obs_dict['top_line_radius'] = info['top_line_radius']
        obs_dict['bottom_line_radius'] = info['bottom_line_radius']
        # obs_dict['touching_screen'] = data.sensordata[self.screen_touch_id] > 0.0

        completed_phase_0 = info['completed_phase_0']
        completed_phase_1 = info['completed_phase_1']
        middle_line_crossed = info['middle_line_crossed']
        ee_pos = obs_dict['fingertip']
        start_line = obs_dict['start_line']
        end_line = obs_dict['end_line']
        bottom_line_radius = obs_dict['bottom_line_radius'][0]
        top_line_radius = obs_dict['top_line_radius'][0]
        path_width = jp.linalg.norm(top_line_radius - bottom_line_radius)
        path_length = 2 * jp.pi * (bottom_line_radius + top_line_radius) / 2
        dist_to_start_line = jp.linalg.norm(ee_pos - start_line, axis=-1)

        ee_pos_rel = ee_pos - obs_dict['screen_pos']
        ee_pos_vec = middle_line_crossed * ee_pos_rel[1:] + (1 - middle_line_crossed) * jp.array([jp.maximum(ee_pos_rel[1], 0.001), ee_pos_rel[2]])
        theta_now = jp.atan2(ee_pos_vec[0], ee_pos_vec[1])
        remaining_angle = jp.fmod((2*jp.pi - theta_now), (2 * jp.pi))
        radius = (bottom_line_radius + top_line_radius) / 2
        path_length_remaining = radius * remaining_angle

        obs_dict["remaining_angle"] = remaining_angle
        obs_dict['percentage_of_remaining_path'] = path_length_remaining / path_length

        # Update phase immediately based on current position
        touching_screen = 1.0 *(jp.linalg.norm(ee_pos[0] - start_line[0]) <= 0.01)
        dist_to_circle_center = jp.linalg.norm(ee_pos[1:3] - obs_dict['screen_pos'][1:3], axis=-1)
        within_path_limits = 1.0 * (dist_to_circle_center <= top_line_radius) * (dist_to_circle_center >= bottom_line_radius)
        within_y_dist = 1.0 * (jp.linalg.norm(ee_pos[1] - start_line[1]) <= 0.01)
        phase_0_completed_now = touching_screen * within_path_limits * within_y_dist
        completed_phase_0 = completed_phase_0 + (1. - completed_phase_0) * phase_0_completed_now

        start_line = obs_dict['start_line']        # Shape: (1024, 3)
        middle_line_pos = jp.stack([start_line[0],start_line[1], obs_dict['screen_pos'][2]-radius], axis=-1)  # (1024, 2)
        close_to_middle_line = (jp.abs(ee_pos[2] - middle_line_pos[2]) <= (path_width/2 + 0.01)) * (jp.abs(ee_pos[1] - middle_line_pos[1]) <= 0.02) * completed_phase_0 * touching_screen
        middle_line_crossed = middle_line_crossed + (1. - middle_line_crossed) * close_to_middle_line
        #jax.debug.print("ee_pos: {}, middle_line_pos: {}, screen_pos: {}, start_line: {}", ee_pos[1:3], middle_line_pos[1:3], obs_dict['screen_pos'][1:3], start_line[1:3])
        #jax.debug.print("ee_pos: {}, middle_line_pos: {}, middle_line_crossed: {}", ee_pos[1:3], middle_line_pos[1:3], middle_line_crossed)
        crossed_line_y = 1.0 * (jp.abs(ee_pos[1] - end_line[1]) <= 0.01) * middle_line_crossed * (jp.abs(ee_pos[2] - end_line[2]) <= path_width/2)
        phase_1_x_dist = jp.linalg.norm(ee_pos[0] - end_line[0])
        touching_screen_phase_1 = 1.0 * (phase_1_x_dist <= 0.01)
        phase_1_completed_now = completed_phase_0 * crossed_line_y * touching_screen_phase_1# * within_path_limits
        completed_phase_1 = completed_phase_1 + (1. - completed_phase_1) * phase_1_completed_now  

        dist_circle_center_to_circle_path_center = bottom_line_radius + path_width/2
        dist_ee_to_circle_path_center = jp.linalg.norm(ee_pos[1:3] - obs_dict['screen_pos'][1:3])
        rel_position = jp.linalg.norm(dist_circle_center_to_circle_path_center - dist_ee_to_circle_path_center)
        softcons_for_bounds = jp.clip(jp.linalg.norm(rel_position) / (path_width / 2), 0, 1) ** 15
        
        # Reset phase 0 when phase 1 is done (and episode ends)
        ## TODO: delay this update to the step function, to ensure consistency between different observations (e.g. when defining the reward function)?
        completed_phase_0 = completed_phase_0 * (1. - completed_phase_1)
        #jax.debug.print("crossed_line_y: {}, completed_phase_1: {}, phase_1_completed_now: {}, end_line[2] - path_width/2 {}, ee_pos[2] {}", crossed_line_y, completed_phase_1, phase_1_completed_now, end_line[2] - path_width/2, ee_pos[2])

        obs_dict["con_0_touching_screen"] = touching_screen
        #obs_dict["con_0_1_within_z_limits"] = (1.-within_z_limits) * completed_phase_0
        obs_dict["out_of_bounds"] = (1.-within_path_limits) * completed_phase_0
        obs_dict["con_0_within_y_dist"] = within_y_dist
        obs_dict["completed_phase_0"] = completed_phase_0
        obs_dict["middle_line_crossed"] = middle_line_crossed
        obs_dict["con_1_crossed_line_y"] = crossed_line_y
        obs_dict["con_1_touching_screen"] = touching_screen_phase_1
        obs_dict["completed_phase_1"] = completed_phase_1
        obs_dict["softcons_for_bounds"] = softcons_for_bounds

        obs_dict["inside_path"] = within_path_limits * touching_screen

        ## Compute distances
        phase_0_distance = dist_to_start_line + path_length
        phase_1_distance = path_length_remaining
        dist = completed_phase_0 * phase_1_distance + (1. - completed_phase_0) * phase_0_distance
        #jax.debug.print("phase_0_distance: {} and dist {} and path_length {}", phase_0_distance, dist, path_length)

        obs_dict["distance_phase_0"] = (1. - completed_phase_0) * phase_0_distance
        obs_dict["distance_phase_1"] = completed_phase_0 * phase_1_distance
        obs_dict["dist"] = dist
        obs_dict["phase_1_x_dist"] = phase_1_x_dist
        obs_dict["path_length_remaining"] = completed_phase_0 * path_length_remaining + (1 - completed_phase_0) * path_length

        ## Additional observations
        obs_dict['target'] = completed_phase_0 * obs_dict['end_line'] + (1. - completed_phase_0) * obs_dict['start_line']
        obs_dict["completed_phase_0_first"] = (1. - info["completed_phase_0"]) * (obs_dict["completed_phase_0"])
        obs_dict["completed_phase_1_first"] = (1. - info["completed_phase_1"]) * (obs_dict["completed_phase_1"])

        #jax.debug.print("completed_phase_0: {}, middle_line_crossed: {}, path_length: {}, path_length_remaining: {}, dist {}", completed_phase_0, middle_line_crossed, path_length, path_length_remaining, dist)

        return obs_dict
       
    def update_info(self, info, obs_dict):
        # TODO: is this really needed? can we drop (almost all) info keys?
        info['fingertip'] = obs_dict['fingertip']
        info['completed_phase_0'] = obs_dict['completed_phase_0']
        info['completed_phase_1'] = obs_dict['completed_phase_1']
        info['middle_line_crossed'] = obs_dict['middle_line_crossed']

        return info
    
    def get_jac_effort_costs(self, obs_dict):
        r_effort = 0.00198*jp.linalg.norm(obs_dict['last_ctrl'])**2 
        r_jacc = 6.67e-6*jp.linalg.norm(obs_dict['qacc'][self._independent_dofs])**2 

        effort_cost = r_effort + r_jacc
        
        return effort_cost
    
    def get_reward_dict(self, obs_dict):#, info):

        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)

        rwd_dict = collections.OrderedDict((
            ('reach',   -1.*(1.-obs_dict['completed_phase_1'])*obs_dict["dist"]), 
            ('bonus_1',   1.*(obs_dict['completed_phase_1'])), 
            ('phase_1_touch',   1.*(obs_dict['completed_phase_0']*(-obs_dict['phase_1_x_dist']) + (1.-obs_dict['completed_phase_0'])*(-0.3))),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            ('jac_effort', -1.* self.get_jac_effort_costs(obs_dict)),
            ('truncated', 1.*obs_dict["out_of_bounds"]),
            ('truncated_progress', 1.*obs_dict["out_of_bounds"]*obs_dict['completed_phase_0']*obs_dict['percentage_of_remaining_path']),
            # # Must keys
            ('done',    1.*(obs_dict['completed_phase_1'])),
            ('bonus_inside_path', 1.*obs_dict['inside_path']),
        ))

        #power_softcons = self.weighted_reward_keys['power_for_softcons']
        #phase_1_tunnel_weight = self.weighted_reward_keys['phase_1_tunnel']

        exclude = {"power_for_softcons", "phase_1_tunnel"}

        rwd_dict['dense'] = jp.sum(
            jp.array([wt * rwd_dict[key] for key, wt in self.weighted_reward_keys.items() if key not in exclude]), axis=0)# + phase_1_tunnel_weight*(1.*(1.-obs_dict['completed_phase_1'])*(obs_dict['completed_phase_0']*(-obs_dict['softcons_for_bounds']**power_softcons) + (1.-obs_dict['completed_phase_0'])*(-0.3)))

        return rwd_dict

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
            'start_line': data.site_xpos[self.start_line_id],
            'end_line': data.site_xpos[self.end_line_id],
        }
    
    def get_custom_tunnel_radii(self, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        rng1, rng2 = jax.random.split(rng, 2)
        inner_radius, outer_radius = self.get_tunnel_limits(rng2, self.min_inner_radius, self.max_inner_radius, self.min_width, self.max_width)
        return inner_radius, outer_radius

    def get_custom_tunnel(self, rng: jax.Array, data: mjx.Data) -> dict[str, jax.Array]:

        inner_radius, outer_radius = self.get_custom_tunnel_radii(rng)
        relevant_positions = self.get_relevant_positions(data)
        tunnel_values = {}
        tunnel_values['bottom_line_radius'] = jp.array([inner_radius])
        tunnel_values['top_line_radius'] = jp.array([outer_radius])

        pos = (outer_radius + inner_radius)/2

        tunnel_values['start_line'] = relevant_positions['screen_pos'] + jp.array([0,0,pos])
        tunnel_values['end_line'] = relevant_positions['screen_pos'] + jp.array([0,0,pos])
        tunnel_values['screen_pos'] = relevant_positions['screen_pos']

        return tunnel_values

    def get_custom_tunnels_steeringlaw(self, rng: jax.Array, screen_pos: jax.Array) -> dict[str, jax.Array]:
        tunnels_total = []
        IDs = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24]

        for ID in IDs:
            combos = 0
            while combos < 3:
                rng, rng2 = jax.random.split(rng, 2)
                W = jax.random.uniform(rng, minval=self.min_width, maxval=self.max_width)
                L = ID * W
                r_inner = (L - jp.pi * W) / (2 * jp.pi)
                if self.min_inner_radius <= r_inner <= self.max_inner_radius:
                    r_outer = r_inner + W
                    pos = (r_inner + r_outer) / 2
                    tunnel_positions = []
                    tunnel_positions.append(jp.array([r_inner, 0, 0]))
                    tunnel_positions.append(jp.array([r_outer, 0, 0]))
                    tunnel_positions.append(screen_pos + jp.array([0, 0, pos]))
                    tunnel_positions.append(screen_pos + jp.array([0, 0, pos]))
                    tunnel_positions.append(screen_pos)
                    combos += 1
                    for i in range(5):
                        tunnels_total.append(tunnel_positions)
                    print(f"Added 5 paths for ID {ID}, L {L}, W {W}, R {pos}")
                
        #print(f"tunnels_total", tunnels_total)
        return tunnels_total

    def reset(self, rng, render_token=None, tunnel_positions=None):

        _, rng = jax.random.split(rng, 2)

        # Reset biomechanical model
        data = self._reset_bm_model(rng)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)

        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                "completed_phase_0": 0.0,
                "completed_phase_1": 0.0,
                "middle_line_crossed": 0.0
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
            'middle_line_crossed': 0.0,
            'dist': 0.0,
            'distance_phase_0': 0.0,
            #'distance_phase_1': 0.0,
            'phase_1_x_dist': 0.0,
            'con_0_touching_screen': 0.0,
            'con_1_touching_screen': 0.0,
            'con_1_crossed_line_y': 0.0,
            #'softcons_for_bounds': 0.0,
            'path_length_remaining': 0.0,
            #'path_length': 0.0,
            'remaining_angle': 0.0,
            'rwd_dist': 0.0,
            'rwd_bonus': 0.0,
            'rwd_touch': 0.0,
            'rwd_jac_effort': 0.0,
            'rwd_truncated_progress': 0.0,
            'rwd_bonus_inside_path': 0.0,
            'out_of_bounds': 0.0,
        }

        return State(data, obs, reward, done, metrics, info)
    
    def auto_reset(self, rng, info_before_reset, **kwargs):
        render_token = info_before_reset["render_token"] if self.vision else None
        return self.reset(rng, render_token=render_token, **kwargs)

    def eval_reset(self, rng, eval_id, **kwargs):
        """Reset function wrapper called by evaluate_policy."""
        _tunnel_position_jp = self.SL_tunnel_positions.at[eval_id].get()
        tunnel_values = {}
        tunnel_values['bottom_line_radius'] = jp.array([_tunnel_position_jp[0][0]])
        tunnel_values['top_line_radius'] = jp.array([_tunnel_position_jp[1][0]])
        tunnel_values['start_line'] = _tunnel_position_jp[2]
        tunnel_values['end_line'] = _tunnel_position_jp[3]
        tunnel_values['screen_pos'] = _tunnel_position_jp[4]

        return self.reset(rng, tunnel_positions=tunnel_values, **kwargs)

    def prepare_eval_rollout(self, rng, **kwargs):
        """Function that can be used to define random parameters to be used across multiple evaluation rollouts/resets.
        May return the number of evaluation episodes that should be rolled out (before this method should be called again)."""
        
        ## Setup evaluation episodes for Steering Law validation
        rng, rng2 = jax.random.split(rng, 2)
        self.SL_tunnel_positions = jp.array(self.get_custom_tunnels_steeringlaw(rng2, screen_pos=jp.array([0.5, -0.27, 0.993])))

        return len(self.SL_tunnel_positions)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
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
        _updated_info = self.update_info(state.info, obs_dict)
        _, _updated_info['rng'] = jax.random.split(rng, 2) #update rng after each step to ensure variability across steps
        state.replace(info=_updated_info)
        #jax.debug.print("path_length_remaining: {}, path_length: {}, radius {}", obs_dict["path_length_remaining"], (obs_dict["top_line_radius"][0] + obs_dict["bottom_line_radius"][0]) * jp.pi, (obs_dict["top_line_radius"][0] + obs_dict["bottom_line_radius"][0])/2)

        done = rwd_dict['done']
        state.metrics.update(
            completed_phase_0 = obs_dict["completed_phase_0"],
            completed_phase_1 = obs_dict["completed_phase_1"],
            dist = obs_dict["dist"],
            distance_phase_0 = obs_dict["distance_phase_0"],
            #distance_phase_1 = obs_dict["distance_phase_1"],
            phase_1_x_dist = obs_dict["phase_1_x_dist"],
            con_0_touching_screen = obs_dict["con_0_touching_screen"],
            con_1_touching_screen = obs_dict["con_1_touching_screen"],
            con_1_crossed_line_y = obs_dict["con_1_crossed_line_y"],
            # softcons_for_bounds = obs_dict["softcons_for_bounds"],
            middle_line_crossed = obs_dict["middle_line_crossed"],
            path_length_remaining = obs_dict["path_length_remaining"],
            #path_length = (obs_dict["top_line_radius"][0] + obs_dict["bottom_line_radius"][0]) * jp.pi,  #2*pi*radius
            remaining_angle = obs_dict["remaining_angle"],
            rwd_dist = rwd_dict['reach']*self.weighted_reward_keys['reach'],
            rwd_bonus = rwd_dict['bonus_1']*self.weighted_reward_keys['bonus_1'],
            rwd_touch = rwd_dict['phase_1_touch']*self.weighted_reward_keys['phase_1_touch'],
            rwd_truncated_progress = rwd_dict['truncated_progress']*self.weighted_reward_keys['truncated_progress'],
            rwd_jac_effort = rwd_dict["jac_effort"]*self.weighted_reward_keys['jac_effort'],
            rwd_bonus_inside_path = rwd_dict["bonus_inside_path"]*self.weighted_reward_keys['bonus_inside_path'], 
            out_of_bounds=obs_dict["out_of_bounds"],
        )
        #jax.debug.print("path_length_remaining: {}, remaining_angle: {}, fingertip_rel {}", obs_dict["path_length_remaining"], obs_dict["remaining_angle"], obs_dict["fingertip"] - obs_dict['screen_pos'])

        #jax.debug.print("rwd_dist: {} and dist {}", rwd_dict['reach']*self.weighted_reward_keys['reach'], obs_dict["dist"])

        return state.replace(
            data=data, obs=obs, reward=rwd_dict['dense'], done=done
        )
    
    # def add_target_pos_to_data(self, data, target_pos):
    #     xpos = data.xpos
    #     geom_xpos = data.geom_xpos

    #     xpos = xpos.at[self.target_body_id].set(target_pos)
    #     geom_xpos = geom_xpos.at[self.target_geom_id].set(target_pos)
    #     data = data.replace(xpos=xpos, geom_xpos=geom_xpos)
    #     return data

    def update_task_visuals(self, mj_model, state):
        screen_pos = state.info["screen_pos"] + jp.array([0.01, 0., 0.])  #need to re-introduce site pos offset from xml file that was ignored in get_custom_tunnel() to ensure that task visuals properly appear in front of the screen 
        mj_model.site_pos[self.start_line_id, :] = state.info["start_line"] - screen_pos - jp.array([0.01, 0., 0.])
        mj_model.site_pos[self.end_line_id, :] = state.info["end_line"] - screen_pos - jp.array([0.01, 0., 0.])

        mj_model.site_size[self.top_line_id, :][0] = state.info["top_line_radius"][0]
        mj_model.site_size[self.bottom_line_id, :][0] = state.info["bottom_line_radius"][0]

        path_width = state.info["top_line_radius"][0] - state.info["bottom_line_radius"][0]

        mj_model.site_size[self.start_line_id, :][2] = path_width/2
        mj_model.site_size[self.end_line_id, :][2] = path_width/2

    def calculate_metrics(self, rollout, eval_metrics_keys={"R^2"}):

        eval_metrics = {}

        if True:
            a,b,r2 = self.calculate_r2(rollout)
            eval_metrics["eval/R^2"] = r2
            eval_metrics["eval/a"] = a
            eval_metrics["eval/b"] = b

        return eval_metrics

    def calculate_r2(self, rollouts, average_r2=True):   
        sl_data = {}
        MTs = np.array([(rollout[np.argwhere(_compl_1)[0].item()].data.time - rollout[np.argwhere(_compl_0)[0].item()].data.time) 
                        for rollout in rollouts if any(_compl_0 := [r.metrics["completed_phase_0"] for r in rollout]) and 
                                                any(_compl_1 := [r.metrics["completed_phase_1"] for r in rollout])])

        Ws = np.array([rollout[np.argwhere(_compl_0)[0].item()].info["top_line_radius"] - rollout[np.argwhere(_compl_0)[0].item()].info["bottom_line_radius"]
            for rollout in rollouts if any(_compl_0 := [r.metrics["completed_phase_0"] for r in rollout]) and 
                                    any(_compl_1 := [r.metrics["completed_phase_1"] for r in rollout])
        ])
        Rs = np.array([(rollout[np.argwhere(_compl_1)[0].item()].info['top_line_radius'] + rollout[np.argwhere(_compl_0)[0].item()].info['bottom_line_radius'])/2
                        for rollout in rollouts if any(_compl_0 := [r.metrics["completed_phase_0"] for r in rollout]) and 
                                                any(_compl_1 := [r.info["completed_phase_1"] for r in rollout])])

        Ds = Rs * (2 * np.pi)

        IDs = (Ds / Ws).reshape(-1, 1)
        
        sl_data.update({'R': Rs})


        if len(IDs) == 0 or len(MTs) == 0:
            return np.nan, np.nan, np.nan
        
        a, b, r2, y_pred, ID_means, MT_means = fit_model(IDs, MTs, average_r2=average_r2)

        print(f"R^2: {r2}, a,b: {a},{b}")

        sl_data.update({"ID": IDs, "MT_ref": MTs,
                "MT_pred": y_pred,
                "D": Ds, "W": Ws})
        if average_r2:
            sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means})

        return a,b,r2
    
def fit_model(IDs, MTs, average_r2):
    if average_r2:
        IDs_rounded = IDs.round(2)
        ID_means = np.sort(np.unique(IDs_rounded)).reshape(-1, 1)
        MT_means = np.array([MTs[np.argwhere(IDs_rounded.flatten() == _id)].mean() for _id in ID_means])

        model = LinearRegression()
        model.fit(ID_means, MT_means)
        a = model.intercept_
        b = model.coef_[0]
        y_pred = model.predict(ID_means)
        r2 = r2_score(MT_means, y_pred)
    else:
        model = LinearRegression()
        model.fit(IDs, MTs)
        a = model.intercept_
        b = model.coef_[0]
        y_pred = model.predict(IDs)
        r2 = r2_score(MTs, y_pred)
    return a, b, r2, y_pred, ID_means, MT_means
