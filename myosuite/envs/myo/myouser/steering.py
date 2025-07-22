from mujoco_playground._src import mjx_env
import abc
from mujoco import mjx
import mujoco
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
import numpy as np


from myosuite.envs.myo.myouser.base import MyoUserBase



class Steering(MyoUserBase):

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

        self.obs_keys =['qpos', 'qvel', 'qacc', 'fingertip', 'act', 'motor_act', 'screen_pos', 'start_line', 'end_line', 'top_line', 'bottom_line'] 


        self._episode_length = self._config.episode_length
        self.screen_id = self._mj_model.geom("screen").id
        self.top_line_id = self._mj_model.site("top_line").id
        self.bottom_line_id = self._mj_model.site("bottom_line").id
        self.start_line_id = self._mj_model.site("start_line").id
        self.end_line_id = self._mj_model.site("end_line").id
        self.fingertip_id = self._mj_model.site("fingertip").id
        self.screen_touch_id = self._mj_model.sensor("screen_touch").id
        #TODO: once contact sensors are integrated, check if the fingertip_geom is needed or not

    def _prepare_after_init(self, data):
        pass

    def get_relevant_positions(self, data: mjx.Data) -> dict[str, jax.Array]:
        return {
            'fingertip': data.site_xpos[self.fingertip_id],
            'screen_pos': data.site_xpos[self.screen_id],
            'top_line': data.site_xpos[self.top_line_id],
            'bottom_line': data.site_xpos[self.bottom_line_id],
            'start_line': data.site_xpos[self.start_line_id],
            'end_line': data.site_xpos[self.end_line_id],
        }

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

        reward, done = jp.zeros(2)
        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                "motor_act": self._motor_act}
        info.update(self.get_relevant_positions(data))
        # obs = self.get_obs(data, info)
        obs, info = self.get_obs_vec(data, info)
        metrics = {
            'success_rate': 0.0,
             'dist': 0.0,
             'touching_screen': 0.0,
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng = state.info['rng']
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)

        data = mjx_env.step(self._mjx_model, state.data, new_ctrl, n_substeps=self.n_substeps)
        
        obs_dict = self.get_obs_dict(data, state.info)
        obs = self.obsdict2obsvec(obs_dict)
        rwd, done, dist = self.get_rewards_and_done(obs_dict)
        _updated_info = self.update_info(state.info, obs_dict)
        _, _updated_info['rng'] = jax.random.split(rng, 2) #update rng after each step to ensure variability across steps
        state.metrics.update(
            success_rate=done,
            dist=dist,
            touching_screen=obs_dict['touching_screen'],
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
        ee_pos = obs_dict['fingertip']
        start_line = obs_dict['start_line']
        diff = ee_pos - start_line
        dist = jp.linalg.norm(diff, axis=-1) # Check of this is correct
        obs_dict['dist'] = dist

        #TODO: actually implmenet properly
        dist = obs_dict['dist']
        reach_reward = (jp.exp(-dist*10) - 1.)/10
        done = 1.0 * (dist <= 0.01)
        success_bonus = 10 * done
        reward = reach_reward + success_bonus
        return reward, done, dist
    
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
        obs_dict['start_line'] = data.site_xpos[self.start_line_id]
        obs_dict['end_line'] = data.site_xpos[self.end_line_id]
        obs_dict['top_line'] = data.site_xpos[self.top_line_id]
        obs_dict['bottom_line'] = data.site_xpos[self.bottom_line_id]
        obs_dict['touching_screen'] = 1.0 * (data.sensordata[self.screen_touch_id] > 0.0)

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

