from mujoco_playground._src import mjx_env
import abc
from mujoco import mjx
import mujoco
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
import numpy as np


from myosuite.envs.myo.myouser.base import MyoUserBase, get_default_config

def steering_config():
    config = get_default_config()
    config['model_path'] = "myosuite/simhive/uitb_sim/steering.xml"
    config['x_reach_metric_coefficient'] = 2.0
    config['x_reach_weight'] = 1.0
    config['success_bonus'] = 50.0
    config['phase_0_to_1_transition_bonus'] = 0.0
    config['screen_friction'] = 0.1
    return config



class CircularSteering(MyoUserBase):

    def __init__(self, config: config_dict.ConfigDict = steering_config(),
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config, config_overrides)

    def modify_mj_model(self, mj_model):
        mj_model.geom('screen').friction = self._config.screen_friction
        mj_model.geom('fingertip_contact').friction = self._config.screen_friction
        return mj_model
        
    def _setup(self):
        zero = jp.zeros(1)

        # Define training properties (sex might be used for fatigue model)
        self.frame_skip = self._config.frame_skip
        self.muscle_condition = self._config.muscle_condition
        self.sex = self._config.sex

        # Define a maximum number of trials per episode (if needed for e.g. evaluation / visualisation)
        self.max_trials = self._config.max_trials
        
        # Define signal-dependent noise
        self.sigdepnoise_type = self._config.noise_params.sigdepnoise_type
        self.sigdepnoise_level = self._config.noise_params.sigdepnoise_level * jp.ones(1)
        self.sigdepnoise_acc = zero  # only used for red/Brownian noise

        # Define constant (i.e., signal-independent) noise
        self.constantnoise_type = self._config.noise_params.constantnoise_type
        self.constantnoise_level = self._config.noise_params.constantnoise_level * jp.ones(1)
        self.constantnoise_acc = zero  # only used for red/Brownian noise

        self.initializeConditions()

        # Define reset type
        self.reset_type = self._config.reset_type
        ## valid reset types:
        valid_reset_types = ("zero", "epsilon_uniform", "range_uniform", None)
        assert (
            self.reset_type in valid_reset_types
        ), f"Invalid reset type '{self.reset_type} (valid types are {valid_reset_types})."

        self.obs_keys =['qpos', 'qvel', 'qacc', 'fingertip', 'act', 'motor_act', 'screen_pos', 'start_line', 'end_line', 'outer_line_radius', 'inner_line_radius', 'completed_phase_0_arr', 'target'] 


        self._episode_length = self._config.episode_length
        self.screen_id = self._mj_model.geom("screen").id
        self.outer_line_id = self._mj_model.site("outer_line").id
        self.inner_line_id = self._mj_model.site("inner_line").id
        self.start_line_id = self._mj_model.site("start_line").id
        self.end_line_id = self._mj_model.site("end_line").id
        self.fingertip_id = self._mj_model.site("fingertip").id
        self.screen_touch_id = self._mj_model.sensor("screen_touch").id
        #TODO: once contact sensors are integrated, check if the fingertip_geom is needed or not

        self.distance_reach_metric_coefficient = self._config.distance_reach_metric_coefficient
        self.x_reach_metric_coefficient = self._config.x_reach_metric_coefficient
        self.x_reach_weight = self._config.x_reach_weight
        self.success_bonus = self._config.success_bonus
        self.phase_0_to_1_transition_bonus = self._config.phase_0_to_1_transition_bonus


        # Currently hardcoded
        self.min_width = 0.2
        self.min_inner_radius = 0.05
        self.max_inner_radius = 0.3
        self.center = 0.0
        self.inner_radius = 0.1
        self.outer_radius = 0.2


    @staticmethod
    def get_tunnel_limits(rng, low, high, min_size):
        rng1, rng2 = jax.random.split(rng, 2)
        small_low = low
        small_high = high - min_size
        small_line = jax.random.uniform(rng1) * (small_high - small_low) + small_low
        large_low = small_line + min_size
        large_high = high
        large_line = jax.random.uniform(rng2) * (large_high - large_low) + large_low
        return small_line, large_line

    def get_custom_tunnel_radii(self, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        rng1, rng2 = jax.random.split(rng, 2)
        inner_radius, outer_radius = self.get_tunnel_limits(rng2, self.min_inner_radius, self.max_inner_radius, self.min_width)

        inner_radius = 0.1
        outer_radius = 0.3
        return inner_radius, outer_radius
    

    def get_custom_tunnel(self, rng: jax.Array, data: mjx.Data) -> dict[str, jax.Array]:

        inner_radius, outer_radius = self.get_custom_tunnel_radii(rng)
        relevant_positions = self.get_relevant_positions(data)
        tunnel_radii = {}
        tunnel_radii['inner_line_radius'] = inner_radius
        tunnel_radii['outer_line_radius'] = outer_radius
        return tunnel_radii
    
    def add_custom_tunnel_to_data(self, data: mjx.Data, tunnel_radii: dict[str, jax.Array]) -> mjx.Data:
        ids = [self.inner_line_id, self.outer_line_id, self.start_line_id, self.end_line_id]
        labels = ['inner_line', 'outer_line']
        for id, label in zip(ids, labels):
            data = data.replace(
                site_size=data.site_size.at[id].set(tunnel_radii[label])
            )
        return data

    def _prepare_after_init(self, data):
        pass

    def get_relevant_positions(self, data: mjx.Data) -> dict[str, jax.Array]:
        return {
            'fingertip': data.site_xpos[self.fingertip_id],
            'screen_pos': data.site_xpos[self.screen_id],
            'outer_line': data.site_xpos[self.outer_line_id],
            'inner_line': data.site_xpos[self.inner_line_id],
            'start_line': data.site_xpos[self.start_line_id],
            'end_line': data.site_xpos[self.end_line_id],
        }
    
    def get_tunnel_positions_from_info(self, info: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {
            'inner_line_radius': info['inner_line_radius'],
            'outer_line_radius': info['outer_line_radius'],
        }

    def prep_rendering_special_tunnel(self, tunnel_radii: dict[str, jax.Array]) -> None:
        assert not self.vision, "Vision is not supported for this type of function!"

        width = (tunnel_radii['outer_line'] - tunnel_radii['inner_line'])
        length = 2 * jp.pi * tunnel_radii['inner_line']
        
        self.mj_model.site('inner_line').size[1] = tunnel_radii['inner_line']
        self.mj_model.site('outer_line').size[1] = tunnel_radii['outer_line']

        self.mj_model.site('start_line').pos[2] = tunnel_radii['outer_line']
        self.mj_model.site('start_line').size[2] = width / 2

        self.mj_model.site('end_line').pos[2] = tunnel_radii['outer_line']
        self.mj_model.site('end_line').size[2] = width / 2


    def reset(self, rng: jax.Array) -> mjx_env.State:
        _, rng = jax.random.split(rng, 2)
        last_ctrl = jp.zeros(self._nu)
        reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
        data = mjx_env.init(
            self._mjx_model,
            qpos=reset_qpos,
            qvel=reset_qvel,
            act=reset_act,
            ctrl=last_ctrl,
        )
        reset_qpos, reset_qvel, reset_act = self._reset_range_uniform(rng, data)
        data = mjx_env.init(
            self._mjx_model,
            qpos=reset_qpos,
            qvel=reset_qvel,
            act=reset_act,
            ctrl=last_ctrl,
        )
        self._reset_bm_model(rng)
        tunnel_radii = self.get_custom_tunnel(rng, data)
        # data = self.add_custom_tunnel_to_data(data, tunnel_positions)

        reward, done = jp.zeros(2)
        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                "motor_act": self._motor_act,
                "completed_phase_0": 0.0}
        info.update(tunnel_radii)
        # info.update(self.get_relevant_positions(data))
        # obs = self.get_obs(data, info)
        obs, info = self.get_obs_vec(data, info)
        metrics = {
            'success_rate': 0.0,
             'dist': 0.0,
             'touching_screen': jp.bool_(False),
             'completed_phase_0': 0.0,
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng = state.info['rng']
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)

        data = mjx_env.step(self._mjx_model, state.data, new_ctrl, n_substeps=self.n_substeps)
        # tunnel_positions = self.get_tunnel_positions_from_info(state.info)
        # data = self.add_custom_tunnel_to_data(data, tunnel_positions)
        
        # Compute observation dictionary using *old* phase information first
        obs_dict = self.get_obs_dict(data, state.info)

        # Compute rewards and phase transition
        rwd, done, dist, completed_phase_0 = self.get_rewards_and_done(obs_dict)

        # ------------------------------------------------------------------
        # Make the observation *consistent* with the updated phase           
        # ------------------------------------------------------------------
        obs_dict['completed_phase_0'] = completed_phase_0
        obs_dict['completed_phase_0_arr'] = jp.array([completed_phase_0])
        # Update the target (interpolates between start and end lines)
        obs_dict['target'] = completed_phase_0 * obs_dict['end_line'] + (1. - completed_phase_0) * obs_dict['start_line']

        # Now build the observation vector
        obs = self.obsdict2obsvec(obs_dict)

        # Update info with the *new* values so the next step starts in sync
        _updated_info = self.update_info(state.info, obs_dict)
        _, _updated_info['rng'] = jax.random.split(rng, 2) #update rng after each step to ensure variability across steps
        _updated_info['completed_phase_0'] = completed_phase_0

        metric_completed_phase_0 = (1. - done) * completed_phase_0 + done * 1.
        state.metrics.update(
            success_rate=done,
            dist=dist,
            touching_screen=obs_dict['touching_screen'],
            completed_phase_0=metric_completed_phase_0
        )
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=rwd,
            done=done,
            metrics=state.metrics,
            info=_updated_info
        )

    def get_rewards_and_done(self, obs_dict: dict) -> jax.Array:
        completed_phase_0 = obs_dict['completed_phase_0']

        ee_pos = obs_dict['fingertip']
        start_line = obs_dict['start_line']
        end_line = obs_dict['end_line']

        inner_line_radius = obs_dict['inner_line_radius']
        outer_line_radius = obs_dict['outer_line_radius']

        path_width = (outer_line_radius - inner_line_radius)
        path_length = 2 * jp.pi * inner_line_radius

        dist_to_start_line = jp.linalg.norm(ee_pos - start_line, axis=-1)
        dist_to_end_line = jp.linalg.norm(ee_pos[1] - end_line[1])

        # Give some intermediate reward for transitioning from phase 0 to phase 1 but only when finger is touching the
        # phase_0_to_1_transition_bonus = self.phase_0_to_1_transition_bonus * (1. - completed_phase_0) * (dist_to_start_line <= 0.01)

        # Update phase immediately based on current position
        touching_screen = 1.0 *(jp.linalg.norm(ee_pos[0] - start_line[0]) <= 0.01)
        within_z_limits = 1.0 * (ee_pos[2] >= inner_line_z) * (ee_pos[2] <= outer_line_z)
        within_y_dist = 1.0 * (jp.linalg.norm(ee_pos[1] - start_line[1]) <= 0.01)

        phase_0_completed = touching_screen * within_z_limits * within_y_dist
        completed_phase_0 = completed_phase_0 + (1. - completed_phase_0) * phase_0_completed

        phase_0_distance = dist_to_start_line + path_length
        phase_1_distance = dist_to_end_line

        dist = completed_phase_0 * phase_1_distance + (1. - completed_phase_0) * phase_0_distance
        d_coef = self.distance_reach_metric_coefficient
        dist_reward = - d_coef * dist # (jp.exp(-dist*d_coef) - 1.)/d_coef

        phase_1_x_dist = jp.linalg.norm(ee_pos[0] - end_line[0])
        x_weight = self.x_reach_weight
        # If in phase 1 and phase_1_x_dist is greater than 0.01, give a reward of -phase_1_x_dist * x_weight
        # phase_1_x_reward = completed_phase_0 * (phase_1_x_dist >= 0.01) * x_weight * (-phase_1_x_dist)

        phase_1_x_reward = completed_phase_0 * x_weight * (-phase_1_x_dist) + (1 - completed_phase_0) * x_weight * (-0.5)

        # phase_1_z_reward = completed_phase_0 * (1. - within_z_limits) * -2.0
        softcons_limit = 3
        relative_position = jp.linalg.norm(inner_line_z - ee_pos[2])
        softcons_for_bounds = softcons_limit * jp.clip(jp.abs(relative_position) / (path_width / 2), 0, 1) ** 15
        phase_1_z_reward = completed_phase_0 * (-softcons_for_bounds) + (1 - completed_phase_0) * (-softcons_limit)

        crossed_line_y = 1.0 * (ee_pos[1] <= end_line[1])
        touching_screen = 1.0 * (phase_1_x_dist <= 0.01)

        done = completed_phase_0 * crossed_line_y * touching_screen * within_z_limits
        
        reward_success = self.success_bonus 
        reward_no_sucess = dist_reward + phase_1_x_reward + phase_1_z_reward
        reward = done * reward_success + (1 - done) * reward_no_sucess

        # Reset phase only when episode ends
        completed_phase_0 = completed_phase_0 * (1. - done)

        return reward, done, dist, completed_phase_0
    
    
    def get_obs_dict(self, data: mjx.Data, info: dict) -> jax.Array:
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
        obs_dict['last_ctrl'] = info['last_ctrl']

        # Smoothed average of motor actuation (only for motor actuators)  #; normalise
        obs_dict['motor_act'] = info['motor_act']  #(self._motor_act.copy() - 0.5) * 2

        # End-effector and target position - read current position from data instead of cached info
        obs_dict['fingertip'] = data.site_xpos[self.fingertip_id]
        obs_dict['screen_pos'] = data.site_xpos[self.screen_id]

        # Read positions directly from current data instead of stale info
        obs_dict['start_line'] = info['start_line']
        obs_dict['end_line'] = info['end_line']
        obs_dict['outer_line_radius'] = info['outer_line_radius']
        obs_dict['inner_line_radius'] = info['inner_line_radius']
        obs_dict['touching_screen'] = data.sensordata[self.screen_touch_id] > 0.0
        obs_dict['completed_phase_0'] = info['completed_phase_0']
        obs_dict['completed_phase_0_arr'] = jp.array([info['completed_phase_0']])

        completed_phase_0 = info['completed_phase_0']
        obs_dict['target'] = completed_phase_0 * obs_dict['end_line'] + (1. - completed_phase_0) * obs_dict['start_line']

        #TODO: more things to add here
        return obs_dict
    
    def obsdict2obsvec(self, obs_dict) -> jax.Array:
        obsvec =jp.concatenate([obs_dict[key] for key in self.obs_keys])
        return {"proprioception": obsvec}

    def get_obs_vec(self, data: mjx.Data, info: dict) -> jax.Array:
        obs_dict = self.get_obs_dict(data, info)
        obs = self.obsdict2obsvec(obs_dict)
        _updated_info = self.update_info(info, obs_dict)
        return obs, _updated_info

    def update_info(self, info, obs_dict):
        info['last_ctrl'] = obs_dict['last_ctrl']
        info['motor_act'] = obs_dict['motor_act']
        info['fingertip'] = obs_dict['fingertip']
        info['touching_screen'] = obs_dict['touching_screen']
        return info
    
    def get_ctrl(self, state: mjx_env.State, action: jp.ndarray, rng: jp.ndarray):
        new_ctrl = action.copy()

        data0 = state.data
        _selected_motor_control = jp.clip(
            state.info["motor_act"] + action[: self._nm], 0, 1
        )
        _selected_muscle_control = jp.clip(
            data0.act[self._muscle_actuators] + action[self._nm :], 0, 1
        )

        _selected_motor_control = jp.clip(action[: self._nm], 0, 1)
        _selected_muscle_control = jp.clip(action[self._nm :], 0, 1)

        if self.sigdepnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.sigdepnoise_type == "white":
                _added_noise = (
                    self.sigdepnoise_level * _selected_muscle_control * _noise
                )
                _selected_muscle_control += _added_noise
            elif self.sigdepnoise_type == "whiteonly":  # only for debugging purposes
                _selected_muscle_control = (
                    self.sigdepnoise_level * _selected_muscle_control * _noise
                )
            elif self.sigdepnoise_type == "red":
                # self.sigdepnoise_acc *= 1 - 0.1
                self.sigdepnoise_acc += (
                    self.sigdepnoise_level * _selected_muscle_control * _noise
                )
                _selected_muscle_control += self.sigdepnoise_acc
            else:
                raise NotImplementedError(f"{self.sigdepnoise_type}")
        if self.constantnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.constantnoise_type == "white":
                _selected_muscle_control += self.constantnoise_level * _noise
            elif self.constantnoise_type == "whiteonly":  # only for debugging purposes
                _selected_muscle_control = self.constantnoise_level * _noise
            elif self.constantnoise_type == "red":
                self.constantnoise_acc += self.constantnoise_level * _noise
                _selected_muscle_control += self.constantnoise_acc
            else:
                raise NotImplementedError(f"{self.constantnoise_type}")

        # Update smoothed online estimate of motor actuation
        self._motor_act = (1 - self._motor_alpha) * state.info[
            "motor_act"
        ] + self._motor_alpha * jp.clip(_selected_motor_control, 0, 1)

        new_ctrl = new_ctrl.at[self._motor_actuators].set(
            self._mj_model.actuator_ctrlrange[self._motor_actuators, 0]
            + self._motor_act
            * (
                self._mj_model.actuator_ctrlrange[self._motor_actuators, 1]
                - self._mj_model.actuator_ctrlrange[self._motor_actuators, 0]
            )
        )
        new_ctrl = new_ctrl.at[self._muscle_actuators].set(
            jp.clip(_selected_muscle_control, 0, 1)
        )

        isNormalized = False  # TODO: check whether we can integrate the default normalization from BaseV0.step

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            _ctrl_after_fatigue, _, _ = self.muscle_fatigue.compute_act(
                new_ctrl[self._muscle_actuators]
            )
            new_ctrl = new_ctrl.at[self._muscle_actuators].set(_ctrl_after_fatigue)
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            new_ctrl = new_ctrl.at[self.EPLpos].set(new_ctrl[self.EIPpos].copy())
            # Set EIP to 0
            new_ctrl = new_ctrl.at[self.EIPpos].set(0)

        return new_ctrl

class Steering_Cheat(Steering):

    def _reset_range_uniform(self, rng, data):
        qpos = jp.array([-4.4993529e-04, -2.0891316e-01,  8.8479556e-02, -8.8497005e-02,
        2.0891567e-01, -4.2286020e-02,  3.4187350e-01,  1.5365790e-01,
       -1.5366049e-01, -3.4186423e-01,  4.2300060e-02,  1.8401126e+00,
        8.6331731e-01, -1.8401110e+00,  3.5031942e-01,  6.8144262e-01,
       -4.0263432e-01])
        qvel = jp.zeros_like(qpos)
        act = jp.array([0.736914, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.491236  , 0.062633  ,
       0.318566  , 0.        , 0.231446  , 0.77142197, 0.868577  ,
       0.005275  , 0.335965  , 0.636831  , 0.158622  , 0.        ,
       0.        , 0.777745  , 0.2798    , 0.        , 0.158079  ,
       0.        ])
        return qpos, qvel, act
    
    def get_rewards_and_done(self, obs_dict: dict) -> jax.Array:
        completed_phase_0 = 1.0
        
        ee_pos = obs_dict['fingertip']
        end_line = obs_dict['end_line']

        dist_to_end_line = jp.linalg.norm(ee_pos - end_line, axis=-1)
        
        dist_reward = (jp.exp(-dist_to_end_line*10) - 1.)/10    
        done = 1.0 * (dist_to_end_line <= 0.01)

        success_bonus = 10. * done
        reward = dist_reward + success_bonus

        return reward, done, dist_to_end_line, completed_phase_0