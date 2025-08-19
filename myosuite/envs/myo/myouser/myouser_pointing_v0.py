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
"""Base class for MyoUser Arm Pointing model."""
import collections
import tqdm

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

@dataclass
class ReachSettings:
    ref_site: str = "humphant"
    target_origin_rel: List[float] = field(default_factory=lambda: [0., 0., 0.])
    target_pos_range: Dict[str, List[List[float]]] = field(default_factory=lambda: {
        "fingertip": [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]],
    })
    target_radius_range: Dict[str, List[float]] = field(default_factory=lambda: {
        "fingertip": [0.05, 0.05],
    })

@dataclass
class PointingTaskConfig:
    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: List[str] = field(default_factory=lambda: [
        "qpos",
        "qvel",
        "qacc",
        "ee_pos",
        "act",
    ])
    omni_keys: List[str] = field(default_factory=lambda: ["target_pos", "target_radius"])
    weighted_reward_keys: Dict[str, float] = field(default_factory=lambda: {
        "reach": 1,
        "bonus": 8,
    })
    max_duration: float = 4.
    max_trials: int = 1
    reset_type: str = "range_uniform"

@dataclass
class PointingEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserPointing"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_reaching_myouser.xml"
    task_config: PointingTaskConfig = field(default_factory=lambda: PointingTaskConfig())

def default_config() -> config_dict.ConfigDict:
    #TODO: update/make use of env_config parameters!
    env_config = config_dict.create(
        # model_path="myosuite/simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml",
        model_path="myosuite/envs/myo/assets/arm/mobl_arms_index_reaching_myouser.xml",
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
            render_width=120,
            render_height=120,
            use_rasterizer=False,
            enabled_geom_groups=[0, 1, 2],
            enabled_cameras=[0],
        ),
        muscle_config=config_dict.create(
            muscle_condition=None,
            sex=None,
            control_type="default",   #"relative"
            noise_params=config_dict.create(
                sigdepnoise_type=None,
                sigdepnoise_level=0.103,
                constantnoise_type=None,
                constantnoise_level=0.185,
            ),
        ),
        
        task_config=config_dict.create(
            reach_settings=config_dict.create(
                ref_site="humphant",
                # ref_site="R.Shoulder_marker",
                target_origin_rel=[0., 0., 0.],
                target_pos_range={
                    "fingertip": [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]],
                    # 'IFtip': [[-0.1, 0.225, -0.3], [0.1, 0.35, 0.3]],
                },
                target_radius_range={
                    "fingertip": [0.05, 0.05],
                    # 'IFtip': [0.05, 0.05],
                },
            ),
            obs_keys=[
                "qpos",
                "qvel",
                "qacc",
                "ee_pos",
                "act",
            ],
            omni_keys=["target_pos", "target_radius"],
            weighted_reward_keys=config_dict.create(
                reach=1,
                bonus=8,
                #neural_effort=0,  #1e-4,
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


class MyoUserPointing(MyoUserBase):    
    def _setup(self):
        """Task specific setup"""
        super()._setup()
        self.reach_settings = self._config.task_config.reach_settings
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

        #TODO: unify target geoms/sites
        self.tip_sids = []
        self.target_sids = []
        sites = self.reach_settings.target_pos_range.keys()
        if sites:
            for site in sites:
                self.tip_sids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
                self.target_sids.append(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target'))
        # self._episode_length = episode_length
        
        self.target_body_id = self.mj_model.body('target').id
        self.target_geom_id = self.mj_model.geom('target_sphere').id

        # Dwelling based selection -- fingertip needs to be inside target for some time
        self.dwell_threshold = 0.25/self.dt  #corresponds to 250ms; for visual-based pointing use 0.5/self.dt; note that self.dt=self._mjx_model.opt.timestep*self.n_substeps
        if self._config.vision.enabled:
            print(f'Using vision, so doubling dwell threshold to {self.dwell_threshold*2}')
            self.dwell_threshold *= 2   

        # Use early termination if target is not hit in time
        self.max_steps_without_hit = self.max_duration/self.dt #corresponds to {max_duration} seconds; note that self.dt=self.mj_model.opt.timestep*self.n_substeps
    
    def _prepare_after_init(self, data):
        super()._prepare_after_init(data)
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.reach_settings.ref_site)].copy() + jp.array(self.reach_settings.target_origin_rel)  #jp.zeros(3,)
    
    # def update_task_visuals(self, target_pos, target_radius):
    #     self.mj_model.body_pos[self.target_body_id, :] = target_pos
    #     self.mj_model.geom_size[self.target_geom_id, 0] = target_radius

    def generate_target_pos(self, rng, target_pos=None):
        # jax.debug.print(f"Generate new target (target area scale={target_area_dynamic_width_scale})")

        # Set target location
        ##TODO: implement _new_target_distance_threshold constraint with rejection sampling!; improve code efficiency (remove for-loop)
        if target_pos is None:
            target_pos = jp.array([])
            # Sample target position
            rng, *rngs = jax.random.split(rng, len(self.reach_settings.target_pos_range)+1)
            for (site, span), _rng in zip(self.reach_settings.target_pos_range.items(), rngs):
                span = jp.array(span)
                # if self.adaptive_task:
                #     span = self.get_current_target_pos_range(span, target_area_dynamic_width_scale)
                # sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target')
                new_position = self.target_coordinates_origin + jax.random.uniform(_rng, shape=self.target_coordinates_origin.shape, minval=span[0], maxval=span[1])
                target_pos = jp.append(target_pos, new_position.copy())
                # self.mj_model.site_pos.at[sid].set(new_position)

        # self._steps_inside_target = jp.zeros(1)

        return target_pos
    
    def generate_target_size(self, rng, target_radius=None):
        # jax.debug.print(f"Generate new target (target area scale={target_area_dynamic_width_scale})")

        # Set target size
        ##TODO: improve code efficiency (remove for-loop)
        if target_radius is None:
            target_radius = jp.array([])
            # Sample target radius
            rng, *rngs = jax.random.split(rng, len(self.reach_settings.target_radius_range)+1)
            for (site, span), _rng in zip(self.reach_settings.target_radius_range.items(), rngs):
                span = jp.array(span)
                # sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target')
                new_radius = jax.random.uniform(_rng, minval=span[0], maxval=span[1])
                target_radius = jp.append(target_radius, new_radius.copy())
                # self.mj_model.site_size[sid][0] = new_radius

        # self._steps_inside_target = jp.zeros(1)

        return target_radius

    def get_current_target_pos_range(self, span, target_area_dynamic_width_scale):
        return target_area_dynamic_width_scale*(span - jp.mean(span, axis=0)) + jp.mean(span, axis=0)

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
        
        # # Smoothed average of motor actuation (only for motor actuators)  #; normalise
        # obs_dict['motor_act'] = info['motor_act']  #(self._motor_act.copy() - 0.5) * 2

        # Store current control input
        obs_dict['last_ctrl'] = data.ctrl.copy()

        # End-effector and target position
        obs_dict['ee_pos'] = jp.vstack([data.site_xpos[self.tip_sids[isite]].copy() for isite in range(len(self.tip_sids))])
        #TODO: decide how to define ee_pos
        obs_dict['target_pos'] = info['target_pos']  #jp.vstack([data.site_xpos[self.target_sids[isite]].copy() for isite in range(len(self.tip_sids))])

        # Distance to target (used for rewards later)
        obs_dict['reach_dist'] = jp.linalg.norm(jp.array(obs_dict['target_pos']) - jp.array(obs_dict['ee_pos']), axis=-1)

        # Target radius
        obs_dict['target_radius'] = info['target_radius']   #jp.array([self.mj_model.site_size[self.target_sids[isite]][0] for isite in range(len(self.tip_sids))])
        # jax.debug.print(f"STEP-Obs: {obs_dict['target_radius']}")
        obs_dict['inside_target'] = jp.squeeze(obs_dict['reach_dist'] < obs_dict['target_radius'])
        # print(obs_dict['inside_target'], jp.ones(1))

        ## we require all end-effector--target pairs to have distance below the respective target radius
        # obs_dict['steps_inside_target'] = (info['steps_inside_target'] + jp.select(obs_dict['inside_target'], jp.ones(1))) * jp.select(obs_dict['inside_target'], jp.ones(1))
        # print(info['steps_inside_target'], jp.select(obs_dict['inside_target'], jp.ones(1)), (info['steps_inside_target'] + jp.select(obs_dict['inside_target'], jp.ones(1))), obs_dict['steps_inside_target'])
        _steps_inside_target = jp.select([obs_dict['inside_target']], [info['steps_inside_target'] + 1], 0)
        _target_timeout = info['steps_since_last_hit'] >= self.max_steps_without_hit
        # print("steps_inside_target", obs_dict['steps_inside_target'])
        obs_dict['target_success'] = _steps_inside_target >= self.dwell_threshold
        obs_dict['target_fail'] = ~obs_dict['target_success'] & _target_timeout
        
        obs_dict['steps_inside_target'] = jp.select([obs_dict['target_success']], [0], _steps_inside_target)
        obs_dict['steps_since_last_hit'] = jp.select([obs_dict['target_success'] | obs_dict['target_fail']], [0], info['steps_since_last_hit'])
        obs_dict['trial_idx'] = info['trial_idx'] + jp.select([obs_dict['target_success'] | obs_dict['target_fail']], jp.ones(1))
        # print("trial_idx", obs_dict['trial_idx'])
        obs_dict['task_completed'] = obs_dict['trial_idx'] >= self.max_trials

        # obs_dict['target_area_dynamic_width_scale'] = info['target_area_dynamic_width_scale']

        # if self.vision:
        #     obs_dict['pixels/view_0'] = info['pixels/view_0']
        #     if self.vision_mode == 'rgb+depth':
        #         obs_dict['pixels/depth'] = info['pixels/depth']
        return obs_dict
    
    def get_obs_vec_aux_task(self, obs_dict, obsvec) -> jp.ndarray:
        target_pos = obs_dict["target_pos"]
        target_radius = obs_dict["target_radius"]
        aux_targets = jp.concatenate([target_pos, target_radius], axis=-1)
        return {
            "proprioception": obsvec,
            "pixels/depth": obs_dict["pixels/depth"],
            "vision_aux_targets": aux_targets,
        }
    
    def update_info(self, info, obs_dict):
        # TODO: is this really needed? can we drop (almost all) info keys?

        ## Update state info of internal variables at the end of each env step
        info['last_ctrl'] = obs_dict['last_ctrl']
        info['steps_since_last_hit'] = obs_dict['steps_since_last_hit']
        info['steps_inside_target'] = obs_dict['steps_inside_target']
        info['trial_idx'] = obs_dict['trial_idx'].copy()
        info['reach_dist'] = obs_dict['reach_dist']

        # Also store variables useful for evaluation
        info['target_success'] = obs_dict['target_success']
        info['target_fail'] = obs_dict['target_fail']
        info['task_completed'] = obs_dict['task_completed']

        return info

    def get_reward_dict(self, obs_dict):
        reach_dist = obs_dict['reach_dist']
        target_radius = obs_dict['target_radius']
        # reach_dist_to_target_bound = jp.linalg.norm(jp.moveaxis(reach_dist-target_radius, -1, -2), axis=-1)
        reach_dist_to_target_bound = jp.linalg.norm(reach_dist-target_radius, axis=-1)
        # steps_inside_target = jp.linalg.norm(obs_dict['steps_inside_target'], axis=-1)
        ctrl_magnitude = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)
        # trial_idx = jp.linalg.norm(obs_dict['trial_idx'], axis=-1)

        # act_mag = jp.linalg.norm(obs_dict['act'], axis=-1)/self._na if self._na != 0 else 0
        # far_th = self.far_th*len(self.tip_sids) if jp.squeeze(obs_dict['time'])>2*self.dt else jp.inf
        # near_th = len(self.tip_sids)*.0125
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   1.*(jp.exp(-reach_dist_to_target_bound*10.) - 1.)/10.),  #-1.*reach_dist)
            ('bonus',   1.*(obs_dict['target_success'])),  #1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('neural_effort', -1.*(ctrl_magnitude ** 2)),
            # ('act_reg', -1.*act_mag),
            # # ('penalty', -1.*(np.any(reach_dist > far_th))),
            # # Must keys
            # ('sparse',  -1.*(jp.linalg.norm(reach_dist, axis=-1) ** 2)),
            ('solved',  1.*(obs_dict['target_success'])),
            ('done',    1.*(obs_dict['task_completed'])), #np.any(reach_dist > far_th))),
        ))
        # print(rwd_dict.items())
        rwd_dict['dense'] = jp.sum(jp.array([wt*rwd_dict[key] for key, wt in self.weighted_reward_keys.items()]), axis=0)

        return rwd_dict

    def reset(self, rng, target_pos=None, target_radius=None, render_token=None):
        # jax.debug.print(f"RESET INIT")

        _, rng = jax.random.split(rng, 2)

        # Reset biomechanical model
        data = self._reset_bm_model(rng)

        # Reset counters
        steps_since_last_hit, steps_inside_target, trial_idx = jp.zeros(3)
        # self._target_success = jp.array(False)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)

        info = {'rng': rng,
                'last_ctrl': last_ctrl,
                'steps_inside_target': steps_inside_target,
                'reach_dist': jp.array(0.),
                'target_success': jp.array(False),
                'steps_since_last_hit': steps_since_last_hit,
                'target_fail': jp.array(False),
                'trial_idx': trial_idx,
                'task_completed': jp.array(False),
                }
        info['target_pos'] = self.generate_target_pos(rng, target_pos=target_pos)
        info['target_radius'] = self.generate_target_size(rng, target_radius=target_radius)
        # if self.vision or self.eval_mode:
        #     self.update_task_visuals(target_pos=info['target_pos'], target_radius=info['target_radius'])
        
        # Generate inital observations
        # TODO: move the following lines into MyoUserBase.reset?
        if self.vision:
            # TODO: do we need to update target information for rendering?
            # data = self.add_target_pos_to_data(data, info["target_pos"])
            info.update(self.generate_pixels(data, render_token=render_token))
        obs, info = self.get_obs_vec(data, info)  #update info from observation made
        # obs_dict = self.get_obs_dict(data, info)
        # obs = self.obsdict2obsvec(obs_dict)

        # self.generate_target(rng, obs_dict)

        reward, done = jp.zeros(2)
        metrics = {'success_rate': 0.,
                    'reach_dist': 0.,
                    'target_success_sum': 0.,  #TODO: remove this and following lines
                    'target_fail_sum': 0.,
                    'target_success_final': 0.,
                    'target_fail_final': 0.,
                }  #'bonus': zero}
        
        return State(data, obs, reward, done, metrics, info)
    
    def auto_reset(self, rng, info_before_reset, **kwargs):
        render_token = info_before_reset["render_token"] if self.vision else None
        return self.reset(rng, render_token=render_token, **kwargs)
    
        # jax.debug.print(f"""RESET WITH CURRICULUM {obs_dict_before_reset["trial_idx"]}, {obs_dict_before_reset["target_radius"]}""")

        rng, rng_reset = jax.random.split(rng, 2)

        # Reset counters
        steps_since_last_hit, steps_inside_target, trial_idx = jp.zeros(3)
        # trial_success_log_pointer_index = jp.zeros(1, dtype=jp.int32)
        # trial_success_log = -1*jp.ones(self.success_log_buffer_length, dtype=jp.int32)
        # self._target_success = jp.array(False)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)  #inserting last_ctrl into pipeline_init is not required, assuming that auto_reset is never called during instatiation of an environment (reset should be used instead)

        # self.robot.sync_sims(self.sim, self.sim_obsd)

        if self.reset_type == "zero":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng_reset)
        elif self.reset_type == "epsilon_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_epsilon_uniform(rng_reset)
        elif self.reset_type == "range_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng_reset)
            data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)
            reset_qpos, reset_qvel, reset_act = self._reset_range_uniform(rng_reset, data)
        else:
            reset_qpos, reset_qvel, reset_act = None, None, None

        data = mjx_env.init(self.mjx_model, qpos=reset_qpos, qvel=reset_qvel, act=reset_act)

        self._reset_bm_model(rng_reset)

        info = {'rng': rng_reset,
                'last_ctrl': last_ctrl,
                'steps_inside_target': steps_inside_target,
                'reach_dist': jp.array(0.),
                'target_success': jp.array(False),
                'steps_since_last_hit': steps_since_last_hit,
                'target_fail': jp.array(False),
                'trial_idx': trial_idx,
                'task_completed': jp.array(False),
                }
        info['target_pos'] = self.generate_target_pos(rng_reset, target_pos=kwargs.get("target_pos", None))
        info['target_radius'] = self.generate_target_size(rng_reset, target_radius=kwargs.get("target_radius", None))
        # if self.vision or self.eval_mode:
        #     self.update_task_visuals(target_pos=info['target_pos'], target_radius=info['target_radius'])

        if self.vision:
            info['render_token'] = info_before_reset['render_token']
            pixels_dict = self.generate_pixels(data, render_token=info['render_token'])
            info.update(pixels_dict)
        obs, info = self.get_obs_vec(data, info)  #update info from observation made
        # obs_dict = self.get_obs_dict(data, info)
        # obs = self.obsdict2obsvec(obs_dict)

        # jax.debug.print(f"obs: {obs}; info-target_radius: {info['target_radius']}")

        # self.generate_target(rng1, obs_dict)

        reward, done = jp.zeros(2)
        metrics = {'success_rate': 0., #obs_dict['success_rate'],
                    'reach_dist': 0.,
                   }  #'bonus': zero}
        
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        # rng = jax.random.PRNGKey(seed=self.seed)  #TODO: fix/move this line, as it does not lead to random perturbations! (generate all random variables in reset function?)
        rng = state.info['rng']

        # increase step counter
        state.info['steps_since_last_hit'] = state.info['steps_since_last_hit'] + 1

        # Generate new target
        ## TODO: can we move the following lines after self.get_ctrl and mjx_env.step are called?
        rng, rng1, rng2 = jax.random.split(rng, 3)
        state.info['target_pos'] = jp.select([(state.info['target_success'] | state.info['target_fail'])], [self.generate_target_pos(rng1)], state.info['target_pos'])
        state.info['target_radius'] = jp.select([(state.info['target_success'] | state.info['target_fail'])], [self.generate_target_size(rng2)], state.info['target_radius'])
        # state.info['target_radius'] = jp.select([(obs_dict['target_success'] | obs_dict['target_fail'])], [jp.array([-151.121])], obs_dict['target_radius']) + jax.random.uniform(rng2)
        # jax.debug.print(f"STEP-Info: {state.info['target_radius']}")

        data0 = state.data
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)
        
        # step forward
        ## TODO: can we move parts of this code into MyoUserBase.step (as a super method)?
        data = mjx_env.step(self.mjx_model, data0, new_ctrl, n_substeps=self.n_substeps)
        # if self.vision or self.eval_mode:
        #     self.update_task_visuals(target_pos=state.info['target_pos'], target_radius=state.info['target_radius'])

        # collect observations and reward
        # obs = self.get_obs_vec(data, state.info)
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
        state.replace(info=_updated_info)
        
        _, state.info['rng'] = jax.random.split(rng, 2)  #update rng after each step to ensure variability across steps

        done = rwd_dict['done']
        state.metrics.update(
            success_rate=done*obs_dict['target_success'],
            reach_dist=done*jp.linalg.norm(obs_dict['reach_dist']),
            # bonus=rwd_dict['bonus'],
            target_success_sum = jp.array(obs_dict['target_success'], dtype=jp.float32),  #TODO: remove this and following lines
            target_fail_sum = jp.array(obs_dict['target_fail'], dtype=jp.float32),
            target_success_final = done*obs_dict['target_success'],
            target_fail_final = done*obs_dict['target_fail'],
        )

        # return self.forward(**kwargs)
        return state.replace(
            data=data, obs=obs, reward=rwd_dict['dense'], done=done
        )
    
    def add_target_pos_to_data(self, data, target_pos):
        xpos = data.xpos
        geom_xpos = data.geom_xpos

        xpos = xpos.at[self.target_body_id].set(target_pos)
        geom_xpos = geom_xpos.at[self.target_geom_id].set(target_pos)
        data = data.replace(xpos=xpos, geom_xpos=geom_xpos)
        return data
    
    def update_task_visuals(self, mj_model, state):
        target_pos = state.info["target_pos"]
        target_radius = state.info["target_radius"]

        mj_model.body_pos[self.target_body_id, :] = target_pos
        mj_model.geom_size[self.target_geom_id, 0] = target_radius.item()