""" =================================================
# Copyright (c) User-in-the-Box 2024; Facebook, Inc. and its affiliates
Authors  :: Florian Fischer (fjf33@cam.ac.uk); Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import collections
import functools

from myosuite.utils import gym
import mujoco
import numpy as np

# from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.fatigue import CumulativeFatigue

import jax
from jax import numpy as jp

from mujoco import mjx

from brax import base
# from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers.training import VmapWrapper
# from brax.mjx.pipeline import _reformat_contact
from brax.io import html, mjcf, model


class LLCEEPosDirectCtrlEnvMJXV0(PipelineEnv):

    # DEFAULT_OBS_KEYS = ['reach_dist', 'inside_target', 'steps_inside_target', 'target_success',    'qpos', 'qvel', 'qacc', 'ee_pos', 'act', 'motor_act', 'target_pos', 'target_radius']  #TODO: exclude 'reach_dist' etc.
    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'qacc', 'ee_pos', 'act', 'motor_act', 'target_pos', 'target_radius']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 8.0,
        #"penalty": 50,
        "neural_effort": 0,  #1e-4,
    }

    def __init__(self, model_path=None, frame_skip=25, # aka physics_steps_per_control_step
            eval_mode=False,
            seed=123, **kwargs):
        # # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # # at the leaf level, when we do inheritance like we do here.
        # # kwargs is needed at the top level to account for injection of __class__ keyword.
        # # Also see: https://github.com/openai/gym/pull/1497
        # gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # # created in __init__ to complete the setup.
        # super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=None)
    
        spec = mujoco.MjSpec.from_file(model_path)

        # b_lunate = spec.body('lunate')
        # b_lunate_pos = b_lunate.pos.copy()

        # # add site to the parent body
        # b_radius = spec.body('radius')
        # b_radius.add_site(
        # name='wrist',
        #     pos=b_lunate_pos,
        #     group=3
        # )

        # # add a target site
        # spec.body('world').add_site(name='wrist_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0.02, 0.02], pos=[-0.2, -0.2, 1.2], rgba=[0, 1, 0, .3])

        mj_model = spec.compile()

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.timestep = 0.002 # dt = mj_model.opt.timestep * frame_skip

        sys = mjcf.load_model(mj_model)
        super().__init__(sys) # **kwargs)
        
        self.seed = seed
        self._n_frames = kwargs.get('n_frames', frame_skip)  #TODO: check that this attribute is used by PipelineEnv
        # self._n_frames = kwargs['n_frames']
        kwargs['backend'] = 'mjx'

        # # self.data = self.pipeline_init(qpos0, qvel0)
        # data = mjx.make_data(sys)  #this is done by self.reset called in self._prepare_env

        self._prepare_env(**kwargs)
    
    def _prepare_env(self, **kwargs):
        self._prepare_bm_model()
        # self._prepare_reaching_task_pt1()
        self._setup(**kwargs)

        # Do a forward step so stuff like geom and body positions are calculated [using MjData rather than mjx.Data, to reduce computational overheat]
        # rng_init = jax.random.PRNGKey(self.seed)
        # init_state = self.reset(rng_init, target_pos=jp.zeros(3))
        # _data = init_state.pipeline_state
        _data = mujoco.MjData(self.sys.mj_model)
        mujoco.mj_forward(self.sys.mj_model, _data)

        self._prepare_after_init(_data)

    def _prepare_bm_model(self):
        # Total number of actuators
        self._nu = self.sys.nu

        # Number of muscle actuators
        self._na = self.sys.na

        # Number of motor actuators
        self._nm = self._nu - self._na
        self._motor_act = jp.zeros((self._nm,))
        self._motor_alpha = 0.9*jp.ones(1)

        # Get actuator names (muscle and motor)
        self._actuator_names = [mujoco.mj_id2name(self.sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.sys.nu)]
        self._muscle_actuator_names = set(np.array(self._actuator_names)[self.sys.actuator_trntype==mujoco.mjtTrn.mjTRN_TENDON])  #model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE
        self._motor_actuator_names = set(self._actuator_names) - self._muscle_actuator_names

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._muscle_actuator_names = sorted(self._muscle_actuator_names, key=self._actuator_names.index)
        self._motor_actuator_names = sorted(self._motor_actuator_names, key=self._actuator_names.index)

        # Find actuator indices in the simulation
        self._muscle_actuators = jp.array([mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                                for actuator_name in self._muscle_actuator_names], dtype=jp.int32)
        self._motor_actuators = jp.array([mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                                for actuator_name in self._motor_actuator_names], dtype=jp.int32)

        # Get joint names (dependent and independent)
        self._joint_names = [mujoco.mj_id2name(self.sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.sys.njnt)]
        self._dependent_joint_names = {self._joint_names[idx] for idx in
                                    np.unique(self.sys.eq_obj1id[self.sys.eq_active0.astype(bool)])} \
        if self.sys.eq_obj1id is not None else set()
        self._independent_joint_names = set(self._joint_names) - self._dependent_joint_names

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._dependent_joint_names = sorted(self._dependent_joint_names, key=self._joint_names.index)
        self._independent_joint_names = sorted(self._independent_joint_names, key=self._joint_names.index)

        # Find dependent and independent joint indices in the simulation
        self._dependent_joints = [mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                for joint_name in self._dependent_joint_names]
        self._independent_joints = [mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                for joint_name in self._independent_joint_names]

        # If there are 'free' type of joints, we'll need to be more careful with which dof corresponds to
        # which joint, for both qpos and qvel/qacc. There should be exactly one dof per independent/dependent joint.
        def get_dofs(joint_indices):
            qpos = jp.array([], dtype=jp.int32)
            dofs = jp.array([], dtype=jp.int32)
            for joint_idx in joint_indices:
                if self.sys.jnt_type[joint_idx] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                    raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                            f"{self._joint_names[joint_idx]} is of type {mujoco.mjtJoint(self.sys.jnt_type[joint_idx]).name}")
                qpos = jp.append(qpos, self.sys.jnt_qposadr[joint_idx])
                dofs = jp.append(dofs, self.sys.jnt_dofadr[joint_idx])
            return qpos, dofs
        self._dependent_qpos, self._dependent_dofs = get_dofs(self._dependent_joints)
        self._independent_qpos, self._independent_dofs = get_dofs(self._independent_joints)

    # def _prepare_reaching_task_pt1(self):
    # 	# Internal variables
    #     zero = jp.zeros(1)

    #     # self._trial_idx = zero  #number of total trials since last reset
    #     # self._targets_hit = zero      
    #     # self.n_hits_adj = zero  #number of successful trials since last target limit adjustment
    #     # self.n_targets_adj = zero  #number of total trials since last target limit adjustment
    #     # self.n_adjs = zero  #number of target limit adjustments
    #     # self.success_rate = zero  #previous success rate


    def _setup(self,
            target_pos_range:dict,
            target_radius_range:dict,
            target_origin_rel:list = jp.zeros(3),  #[0.225, -0.1, 0.05],  #NOTE: target area offset should be directly added to target_pos_range
            ref_site = 'humphant',
            muscle_condition = None,
            sex = None,
            max_trials = 10,
            sigdepnoise_type = None,   #"white"
            sigdepnoise_level = 0.103,
            constantnoise_type = None,   #"white"
            constantnoise_level = 0.185,
            reset_type = "range_uniform",
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            episode_length = 800,
            frame_skip = 25,  #frame_skip=25 corresponds to 20Hz; with episode_length=800, this results in 40s maximum time per episode
            **kwargs,
        ):
        # self.target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + jp.array(target_origin_rel)
        self.target_pos_range = target_pos_range
        self.target_radius_range = target_radius_range
        self.target_origin_rel = target_origin_rel
        self.ref_site = ref_site

        zero = jp.zeros(1)
        
        # Define training properties (sex might be used for fatigue model)
        self.frame_skip = frame_skip
        self.muscle_condition = muscle_condition
        self.sex = sex

        # Define a maximum number of trials per episode (if needed for e.g. evaluation / visualisation)
        self.max_trials = max_trials

        # Define signal-dependent noise
        self.sigdepnoise_type = sigdepnoise_type
        self.sigdepnoise_level = sigdepnoise_level * jp.ones(1)
        self.sigdepnoise_acc = zero  #only used for red/Brownian noise
        
        # Define constant (i.e., signal-independent) noise
        self.constantnoise_type = constantnoise_type
        self.constantnoise_level = constantnoise_level * jp.ones(1)
        self.constantnoise_acc = zero  #only used for red/Brownian noise

        self.initializeConditions()

        # Define reset type
        self.reset_type = reset_type
        ## valid reset types: 
        valid_reset_types = ("zero", "epsilon_uniform", "range_uniform", None)
        assert self.reset_type in valid_reset_types, f"Invalid reset type '{self.reset_type} (valid types are {valid_reset_types})."

        # # Initialise other variables required for setup (i.e., by get_obs_vec, get_obs_dict, get_reward_dict, or get_env_infos)
        # self.dwell_threshold = zero  #0.
        
        # Setup reward function and observation components, and other meta-parameters
        self.rwd_keys_wt = weighted_reward_keys
        self.obs_keys = obs_keys

        self.tip_sids = []
        self.target_sids = []
        sites = self.target_pos_range.keys()
        if sites:
            for site in sites:
                self.tip_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
                self.target_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target'))
        self._episode_length = episode_length
        # self._normalize_act = normalize_act

        #### Task-specific settings (data-dependent settings need to be defined in _prepare_after_init) ####
        # # Define success/fail log buffer and its pointer
        # self._trial_success_log = -1*jp.ones(self.success_log_buffer_length, dtype=jp.int32)  #logging whether trials since last target limit adjustment were successful (2D array, with first entry batch index, and second entry trial_id within that single run)
        # self._trial_success_log_pointer_index = jp.zeros(1, dtype=jp.int32)  #1D array of batch size, where each entry corresponds to the current pointer index, defining for each run where to append the next success/failure boolean
        
        # Dwelling based selection -- fingertip needs to be inside target for some time
        self.dwell_threshold = 0.25/self.dt  #corresponds to 250ms; for visual-based pointing use 0.5/self.dt; note that self.dt=self.sys.opt.timestep*self._n_frames
        
        # Use early termination if target is not hit in time
        # self._steps_since_last_hit = jp.zeros(1)
        self._max_steps_without_hit = 4./self.dt #corresponds to 4 seconds; note that self.dt=self.sys.opt.timestep*self._n_frames
    
    
    def _prepare_after_init(self, data):
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ref_site)].copy() + jp.array(self.target_origin_rel)  #jp.zeros(3,)

    
    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self.sys.mj_model.actuator_gainprm.shape[0]):
                self.sys.mj_model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self.sys.mj_model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.sys.mj_model, frame_skip=self.frame_skip, sex=self.sex, seed=self.get_input_seed()
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self.sys.mj_model.actuator_name2id("EPL")
            self.EIPpos = self.sys.mj_model.actuator_name2id("EIP")

    def get_ctrl(self, state: State, action: jp.ndarray, rng: jp.ndarray):
        new_ctrl = action.copy()

        _selected_motor_control = jp.clip(action[:self._nm], 0, 1)
        _selected_muscle_control = jp.clip(action[self._nm:], 0, 1)

        if self.sigdepnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.sigdepnoise_type == "white":
                _added_noise = self.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += _added_noise
            elif self.sigdepnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.sigdepnoise_level*_selected_muscle_control*_noise
            elif self.sigdepnoise_type == "red":
                # self.sigdepnoise_acc *= 1 - 0.1
                self.sigdepnoise_acc += self.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += self.sigdepnoise_acc
            else:
                raise NotImplementedError(f"{self.sigdepnoise_type}")
        if self.constantnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.constantnoise_type == "white":
                _selected_muscle_control += self.constantnoise_level*_noise
            elif self.constantnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.constantnoise_level*_noise
            elif self.constantnoise_type == "red":
                self.constantnoise_acc += self.constantnoise_level*_noise
                _selected_muscle_control += self.constantnoise_acc
            else:
                raise NotImplementedError(f"{self.constantnoise_type}")

        # # Update smoothed online estimate of motor actuation
        # self._motor_act = (1 - self._motor_alpha) * self._motor_act \
        #                         + self._motor_alpha * np.clip(_selected_motor_control, 0, 1)
        self._motor_act = _selected_motor_control

        new_ctrl = new_ctrl.at[self._motor_actuators].set(self.sys.mj_model.actuator_ctrlrange[self._motor_actuators, 0] + self._motor_act*(self.sys.mj_model.actuator_ctrlrange[self._motor_actuators, 1] - self.sys.mj_model.actuator_ctrlrange[self._motor_actuators, 0]))
        new_ctrl = new_ctrl.at[self._muscle_actuators].set(jp.clip(_selected_muscle_control, 0, 1))

        isNormalized = False  #TODO: check whether we can integrate the default normalization from BaseV0.step
        
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
    
    # step the simulation forward (overrides BaseV0.step; --> also, enable signal-dependent and/or constant motor noise)
    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        # rng = jax.random.PRNGKey(seed=self.seed)  #TODO: fix/move this line, as it does not lead to random perturbations! (generate all random variables in reset function?)
        rng = state.info['rng']

        # increase step counter
        state.info['steps_since_last_hit'] = state.info['steps_since_last_hit'] + 1

        # Generate new target
        ##TODO: update data0 when target is updated?
        rng, rng1, rng2 = jax.random.split(rng, 3)
        state.info['target_pos'] = jp.select([(state.info['target_success'] | state.info['target_fail'])], [self.generate_target_pos(rng1)], state.info['target_pos'])
        state.info['target_radius'] = jp.select([(state.info['target_success'] | state.info['target_fail'])], [self.generate_target_size(rng2)], state.info['target_radius'])
        # state.info['target_radius'] = jp.select([(obs_dict['target_success'] | obs_dict['target_fail'])], [jp.array([-151.121])], obs_dict['target_radius']) + jax.random.uniform(rng2)
        # jax.debug.print(f"STEP-Info: {state.info['target_radius']}")

        data0 = state.pipeline_state
        rng, rng_ctrl = jax.random.split(rng, 2)
        new_ctrl = self.get_ctrl(state, action, rng_ctrl)
        
        # step forward
        # self.last_ctrl = self.robot.step(
        #     ctrl_desired=new_ctrl,
        #     ctrl_normalized=isNormalized,
        #     step_duration=self.dt,
        #     realTimeSim=self.mujoco_render_frames,
        #     render_cbk=self.mj_render if self.mujoco_render_frames else None,
        # )
        # self.last_ctrl = new_ctrl  #TODO: is this required?
        data = self.pipeline_step(data0, new_ctrl)

        # collect observations and reward
        # obs = self.get_obs_vec(data, state.info)
        obs_dict = self.get_obs_dict(data, state.info)
        obs = self.obsdict2obsvec(obs_dict)
        rwd_dict = self.get_reward_dict(obs_dict)

        _updated_info = self.update_info(state.info, obs_dict)
        state.replace(info=_updated_info)

        _, state.info['rng'] = jax.random.split(rng, 2)  #update rng after each step to ensure variability across steps

        state.metrics.update(
            # bonus=rwd_dict['bonus'],
        )

        # return self.forward(**kwargs)
        return state.replace(
            pipeline_state=data, obs=obs, reward=rwd_dict['dense'], done=rwd_dict['done']
        )
    
    # # updates executed at each step, after MuJoCo step (see BaseV0.step) but before MyoSuite returns observations, reward and infos (see MujocoEnv.forward)
    # def _forward(self, **kwargs):
    #     pass
        
    #     # continue with default forward step
    #     super()._forward(**kwargs)

    def get_obs_vec(self, data, info):
        obs_dict = self.get_obs_dict(data, info)
        obs = self.obsdict2obsvec(obs_dict)
        _updated_info = self.update_info(info, obs_dict)
        return obs, _updated_info

    def get_obs_dict(self, data, info):
        rng = info['rng']

        obs_dict = {}
        obs_dict['time'] = jp.array(data.time)
        
        # Normalise qpos
        jnt_range = self.sys.mj_model.jnt_range[self._independent_joints]
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

        # End-effector and target position
        obs_dict['ee_pos'] = jp.vstack([data.site_xpos[self.tip_sids[isite]].copy() for isite in range(len(self.tip_sids))])
        obs_dict['target_pos'] = info['target_pos']  #jp.vstack([data.site_xpos[self.target_sids[isite]].copy() for isite in range(len(self.tip_sids))])

        # Distance to target (used for rewards later)
        obs_dict['reach_dist'] = jp.linalg.norm(jp.array(obs_dict['target_pos']) - jp.array(obs_dict['ee_pos']), axis=-1)

        # Target radius
        obs_dict['target_radius'] = info['target_radius']   #jp.array([self.sys.mj_model.site_size[self.target_sids[isite]][0] for isite in range(len(self.tip_sids))])
        # jax.debug.print(f"STEP-Obs: {obs_dict['target_radius']}")
        obs_dict['inside_target'] = jp.squeeze(obs_dict['reach_dist'] < obs_dict['target_radius'])
        # print(obs_dict['inside_target'], jp.ones(1))

        ## we require all end-effector--target pairs to have distance below the respective target radius
        # obs_dict['steps_inside_target'] = (info['steps_inside_target'] + jp.select(obs_dict['inside_target'], jp.ones(1))) * jp.select(obs_dict['inside_target'], jp.ones(1))
        # print(info['steps_inside_target'], jp.select(obs_dict['inside_target'], jp.ones(1)), (info['steps_inside_target'] + jp.select(obs_dict['inside_target'], jp.ones(1))), obs_dict['steps_inside_target'])
        _steps_inside_target = jp.select([obs_dict['inside_target']], [info['steps_inside_target'] + 1], 0)
        _target_timeout = info['steps_since_last_hit'] >= self._max_steps_without_hit
        # print("steps_inside_target", obs_dict['steps_inside_target'])
        obs_dict['target_success'] = _steps_inside_target >= self.dwell_threshold
        obs_dict['target_fail'] = ~obs_dict['target_success'] & _target_timeout
        
        obs_dict['steps_inside_target'] = jp.select([obs_dict['target_success']], [0], _steps_inside_target)
        obs_dict['steps_since_last_hit'] = jp.select([obs_dict['target_success'] | obs_dict['target_fail']], [0], info['steps_since_last_hit'])
        obs_dict['trial_idx'] = info['trial_idx'] + jp.select([obs_dict['target_success'] | obs_dict['target_fail']], jp.ones(1))
        # print("trial_idx", obs_dict['trial_idx'])
        obs_dict['task_completed'] = obs_dict['trial_idx'] >= self.max_trials

        return obs_dict
    
    def obsdict2obsvec(self, obs_dict) -> jp.ndarray:
        obs_list = [jp.zeros(0)]
        for key in self.obs_keys:
            obs_list.append(obs_dict[key].ravel()) # ravel helps with images
        obsvec = jp.concatenate(obs_list)

        return obsvec
    
    def update_info(self, info, obs_dict):
        ## Update state info of internal variables at the end of each env step
        info['last_ctrl'] = obs_dict['last_ctrl']
        info['motor_act'] = obs_dict['motor_act']
        info['steps_since_last_hit'] = obs_dict['steps_since_last_hit']
        info['steps_inside_target'] = obs_dict['steps_inside_target']
        info['trial_idx'] = obs_dict['trial_idx'].copy()

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
        last_ctrl = jp.linalg.norm(obs_dict['last_ctrl'], axis=-1)
        # trial_idx = jp.linalg.norm(obs_dict['trial_idx'], axis=-1)

        act_mag = jp.linalg.norm(obs_dict['act'], axis=-1)/self._na if self._na != 0 else 0
        # far_th = self.far_th*len(self.tip_sids) if jp.squeeze(obs_dict['time'])>2*self.dt else jp.inf
        # near_th = len(self.tip_sids)*.0125
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   1.*(jp.exp(-reach_dist_to_target_bound*10.) - 1.)/10.),  #-1.*reach_dist)
            ('bonus',   1.*(obs_dict['target_success'])),  #1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('neural_effort', -1.*(last_ctrl ** 2)),
            ('act_reg', -1.*act_mag),
            # ('penalty', -1.*(np.any(reach_dist > far_th))),
            # Must keys
            ('sparse',  -1.*(jp.linalg.norm(reach_dist, axis=-1) ** 2)),
            ('solved',  1.*(obs_dict['target_success'])),
            ('done',    1.*(obs_dict['task_completed'])), #np.any(reach_dist > far_th))),
        ))
        # print(rwd_dict.items())
        rwd_dict['dense'] = jp.sum(jp.array([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()]), axis=0)
        return rwd_dict
    
    def get_env_infos(self, state: State, data: mjx.Data):
        """
        Get information about the environment.
        """
        ## TODO: update this function!!

        # # resolve if current visuals are available
        # if self.visual_dict and "time" in self.visual_dict.keys() and self.visual_dict['time']==self.obs_dict['time']:
        #     visual_dict = self.visual_dict
        # else:
        #     visual_dict = {}

        env_info = {
            'time': self.obs_dict['time'][()],          # MDP(t)
            'rwd_dense': self.rwd_dict['dense'][()],    # MDP(t)
            'rwd_sparse': self.rwd_dict['sparse'][()],  # MDP(t)
            'solved': self.rwd_dict['solved'][()],      # MDP(t)
            'done': self.rwd_dict['done'][()],          # MDP(t)
            'obs_dict': self.obs_dict,                  # MDP(t)
            # 'visual_dict': visual_dict,                 # MDP(t), will be {} if user hasn't explicitly updated self.visual_dict at the current time
            # 'proprio_dict': self.proprio_dict,          # MDP(t)
            'rwd_dict': self.rwd_dict,                  # MDP(t)
            'state': state.pipeline_state,              # MDP(t)
        }

        return env_info

    # generate a valid target
    def generate_target_pos(self, rng, target_pos=None):
        # jax.debug.print(f"Generate new target (target area scale={target_area_dynamic_width_scale})")

        # Set target location
        ##TODO: implement _new_target_distance_threshold constraint with rejection sampling!; improve code efficiency (remove for-loop)
        if target_pos is None:
            target_pos = jp.array([])
            # Sample target position
            rng, *rngs = jax.random.split(rng, len(self.target_pos_range)+1)
            for (site, span), _rng in zip(self.target_pos_range.items(), rngs):
                # sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target')
                new_position = self.target_coordinates_origin + jax.random.uniform(_rng, shape=self.target_coordinates_origin.shape, minval=span[0], maxval=span[1])
                target_pos = jp.append(target_pos, new_position.copy())
                # self.sys.mj_model.site_pos.at[sid].set(new_position)

        # self._steps_inside_target = jp.zeros(1)

        return target_pos
    
    def generate_target_size(self, rng, target_radius=None):
        # jax.debug.print(f"Generate new target (target area scale={target_area_dynamic_width_scale})")

        # Set target size
        ##TODO: improve code efficiency (remove for-loop)
        if target_radius is None:
            target_radius = jp.array([])
            # Sample target radius
            rng, *rngs = jax.random.split(rng, len(self.target_radius_range)+1)
            for (site, span), _rng in zip(self.target_radius_range.items(), rngs):
                # sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target')
                new_radius = jax.random.uniform(_rng, minval=span[0], maxval=span[1])
                target_radius = jp.append(target_radius, new_radius.copy())
                # self.sys.mj_model.site_size[sid][0] = new_radius

        # self._steps_inside_target = jp.zeros(1)

        return target_radius
    
    # def update_adaptive_target_area_width(self, info, obs_dict, success_rate):
    #     # if self.adaptive_change_trial_buffer_length is not None:
    #     #     # Clip success/fail buffer
    #     #     self._trial_success_log = self._trial_success_log[-self.adaptive_change_trial_buffer_length:]

    #     # self.n_targets_adj = jp.sum(self._trial_success_log != 0, axis=-1)
    #     # if self.n_targets_adj >= self.adaptive_change_min_trials:
        
    #     # print((self.adaptive_increase_success_rate, self.target_area_dynamic_width_scale, self.adaptive_change_step_size))
        
        
    #     # self.update_adaptive_target_area_width(new_target_area_width, obs_dict)
    #     # self.n_adjs += 1
    #     # print(f"ADAPTIVE TARGETS -- Adj. #{self.n_adjs}: {self.target_area_dynamic_width_scale*100}% -> {new_target_area_width*100}% (success_rate={self.success_rate})")

    #     # Reset internally used counters
    #     zero = jp.zeros(1)
    #     obs_dict['trial_success_log'].at[:].set(-1) #= jp.zeros([self.success_log_buffer_length], dtype=jp.int32)
    #     obs_dict['trial_success_log_pointer_index'] = zero  #jp.array([], dtype=jp.int32)
    #     # n_hits_adj = zero  #TODO: remove (useless)
    #     # n_targets_adj = zero  #TODO: remove (useless)

    #     # self.target_area_dynamic_width_scale = new_target_area_width
    #     # if (self.success_rate >= self.adaptive_increase_success_rate) and (self.target_area_dynamic_width_scale < 1):
    #     #     new_target_area_width = self.target_area_dynamic_width_scale + self.adaptive_change_step_size
    #     #     self.update_adaptive_target_area_width(new_target_area_width)
    #     # elif (self.success_rate <= self.adaptive_decrease_success_rate) and (self.target_area_dynamic_width_scale > 0):
    #     #     new_target_area_width = self.target_area_dynamic_width_scale - self.adaptive_change_step_size
    #     #     self.update_adaptive_target_area_width(new_target_area_width)
    
    #     # return jp.zeros(1)  #we need to return some object to be able to call this function via jp.select
    #     return obs_dict

    # def update_adaptive_target_area_width(self, new_target_area_width, obs_dict):
    #     self.n_adjs += 1
    #     print(f"ADAPTIVE TARGETS -- Adj. #{self.n_adjs}: {self.target_area_dynamic_width_scale*100}% -> {new_target_area_width*100}% (success_rate={self.success_rate})")

    #     # Reset internally used counters
    #     zero = jp.zeros(1)
    #     obs_dict['trial_success_log'].at[:].set(-1) #= jp.zeros([self.success_log_buffer_length], dtype=jp.int32)
    #     obs_dict['trial_success_log_pointer_index'] = zero  #jp.array([], dtype=jp.int32)
    #     n_hits_adj = zero  #TODO: remove (useless)
    #     n_targets_adj = zero  #TODO: remove (useless)

    #     self.target_area_dynamic_width_scale = new_target_area_width

    def reset(self, rng, **kwargs):
        # jax.debug.print(f"RESET INIT")

        _, rng = jax.random.split(rng, 2)

        # Reset counters
        steps_since_last_hit, steps_inside_target, trial_idx = jp.zeros(3)
        # self._target_success = jp.array(False)
        
        # Reset last control (used for observations only)
        last_ctrl = jp.zeros(self._nu)

        # self.robot.sync_sims(self.sim, self.sim_obsd)

        if self.reset_type == "zero":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
        elif self.reset_type == "epsilon_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_epsilon_uniform(rng)
        elif self.reset_type == "range_uniform":
            reset_qpos, reset_qvel, reset_act = self._reset_zero(rng)
            data = self.pipeline_init(reset_qpos, reset_qvel, act=reset_act, ctrl=last_ctrl)
            reset_qpos, reset_qvel, reset_act = self._reset_range_uniform(rng, data)
        else:
            reset_qpos, reset_qvel, reset_act = None, None, None

        data = self.pipeline_init(reset_qpos, reset_qvel, act=reset_act, ctrl=last_ctrl)
        
        self._reset_bm_model(rng)

        info = {'last_ctrl': last_ctrl,
                'motor_act': self._motor_act,
                'steps_since_last_hit': steps_since_last_hit,
                'steps_inside_target': steps_inside_target,
                'trial_idx': trial_idx,
                'rng': rng,
                'target_success': jp.array(False),
                'target_fail': jp.array(False),
                'task_completed': jp.array(False),
                }
        info['target_pos'] = self.generate_target_pos(rng, target_pos=kwargs.get("target_pos", None))
        info['target_radius'] = self.generate_target_size(rng, target_radius=kwargs.get("target_radius", None))
        obs, info = self.get_obs_vec(data, info)  #update info from observation made
        # obs_dict = self.get_obs_dict(data, info)
        # obs = self.obsdict2obsvec(obs_dict)

        # self.generate_target(rng, obs_dict)

        reward, done = jp.zeros(2)
        metrics = {
                }  #'bonus': zero}
        
        return State(data, obs, reward, done, metrics, info)
    
    def _reset_zero(self, rng):
        """ Resets the biomechanical model. """

        # Set joint angles and velocities to zero
        rng, rng1 = jax.random.split(rng, 2)
        nqi = len(self._independent_qpos)
        qpos = jp.zeros((nqi,))
        qvel = jp.zeros((nqi,))
        reset_qpos = jp.zeros((self.sys.mj_model.nq,))
        reset_qvel = jp.zeros((self.sys.mj_model.nv,))

        # Randomly sample act within unit interval
        reset_act = jax.random.uniform(rng1, shape=self._na, minval=jp.zeros((self._na,)), maxval=jp.ones((self._na,)))

        zero = jp.zeros(1)

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._dependent_qpos].set(zero)
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        reset_qvel = reset_qvel.at[self._dependent_dofs].set(zero)
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)

        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act
    
    def _reset_epsilon_uniform(self, rng):
        """ Resets the biomechanical model. """

        # Randomly sample qpos and qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        qpos = jax.random.uniform(rng1, shape=nqi, minval=jp.ones((nqi,))*-0.05, maxval=jp.ones((nqi,))*0.05)
        qvel = jax.random.uniform(rng2, shape=nqi, minval=jp.ones((nqi,))*-0.05, maxval=jp.ones((nqi,))*0.05)
        reset_qpos = jp.zeros((self.sys.mj_model.nq,))
        reset_qvel = jp.zeros((self.sys.mj_model.nv,))
        reset_act = jax.random.uniform(rng3, shape=self._na, minval=jp.zeros((self._na,)), maxval=jp.ones((self._na,)))

        zero = jp.zeros(1)

        # Set qpos and qvel
        ## TODO: ensure that constraints are initially satisfied
        reset_qpos = reset_qpos.at[self._dependent_qpos].set(zero)
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        reset_qvel = reset_qvel.at[self._dependent_dofs].set(zero)
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)
        
        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def _reset_range_uniform(self, rng, data):
        """ Resets the biomechanical model. """

        # Randomly sample qpos within joint range, qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        jnt_range = self.sys.mj_model.jnt_range[self._independent_joints]
        qpos = jax.random.uniform(rng1, shape=(nqi,), minval=jnt_range[:, 0], maxval=jnt_range[:, 1])
        qvel = jax.random.uniform(rng2, shape=(nqi,), minval=jp.ones((nqi,))*-0.05, maxval=jp.ones((nqi,))*0.05)
        reset_qpos = jp.zeros((self.sys.mj_model.nq,))
        reset_qvel = jp.zeros((self.sys.mj_model.nv,))
        reset_act = jax.random.uniform(rng3, shape=self._na, minval=jp.zeros((self._na,)), maxval=jp.ones((self._na,)))

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        # reset_qpos[self._dependent_qpos] = 0
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)
        # reset_qvel[self._dependent_dofs] = 0
        self.ensure_dependent_joint_angles(data)
        
        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def ensure_dependent_joint_angles(self, data):
        """ Adjusts virtual joints according to active joint constraints. """

        _joint_constraints = self.sys.mj_model.eq_type == 2
        _active_eq_constraints = data.eq_active == 1

        for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
                self.sys.mj_model.eq_obj1id[_joint_constraints & _active_eq_constraints],
                self.sys.mj_model.eq_obj2id[_joint_constraints & _active_eq_constraints],
                self.sys.mj_model.eq_data[_joint_constraints & _active_eq_constraints, 4::-1]):
            if physical_joint_id >= 0:
                new_qpos = data.qpos  #TODO: copy required?
                new_qpos = new_qpos.at[virtual_joint_id].set(jp.polyval(poly_coefs, data.qpos[physical_joint_id]))  #TODO: check mapping between joints (njnt) and dofs (nq/nv)
                data.replace(qpos=new_qpos)

    def _reset_bm_model(self, rng):
        #TODO: do not store anything in self in this function, as its values should mostly be discarded after it is called (no permanent env changes!)

        # Sample random initial values for motor activation
        rng, rng1 = jax.random.split(rng, 2)
        self._motor_act = jax.random.uniform(rng1, shape=(self._nm,), minval=jp.zeros((self._nm,)), maxval=jp.ones((self._nm,)))
        # Reset smoothed average of motor actuator activation
        self._motor_smooth_avg = jp.zeros((self._nm,))

        # Reset accumulative noise
        zero = jp.zeros(1)
        self._sigdepnoise_acc = zero
        self._constantnoise_acc = zero
    

    def render(
        self,
        trajectory: List[base.State],
        target_pos: Optional[jp.ndarray] = None,
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
    ) -> Sequence[np.ndarray]:
        """Sets target positions and then renders a trajectory using the MuJoCo renderer."""
        if target_pos is not None:
            for _target_sid, _target_pos in zip(self.target_sids, jp.atleast_2d(target_pos)):
                self.sys.mj_model.site(_target_sid).pos = _target_pos
                # self.sys = self.sys.tree_replace({'site_pos': self.sys.mj_model.site_pos})
                # print(self.sys.site_pos[_target_sid])#.set(new_position)

        return super().render(trajectory, height, width, camera)

class LLCEEPosEnvMJXV0(LLCEEPosDirectCtrlEnvMJXV0):
    def get_ctrl(self, state: State, action: jp.ndarray, rng: jp.ndarray):
        new_ctrl = action.copy()

        data0 = state.pipeline_state
        _selected_motor_control = jp.clip(state.info['motor_act'] + action[:self._nm], 0, 1)
        _selected_muscle_control = jp.clip(data0.act[self._muscle_actuators] + action[self._nm:], 0, 1)

        _selected_motor_control = jp.clip(action[:self._nm], 0, 1)
        _selected_muscle_control = jp.clip(action[self._nm:], 0, 1)

        if self.sigdepnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.sigdepnoise_type == "white":
                _added_noise = self.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += _added_noise
            elif self.sigdepnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.sigdepnoise_level*_selected_muscle_control*_noise
            elif self.sigdepnoise_type == "red":
                # self.sigdepnoise_acc *= 1 - 0.1
                self.sigdepnoise_acc += self.sigdepnoise_level*_selected_muscle_control*_noise
                _selected_muscle_control += self.sigdepnoise_acc
            else:
                raise NotImplementedError(f"{self.sigdepnoise_type}")
        if self.constantnoise_type is not None:
            rng, rng1 = jax.random.split(rng, 2)
            _noise = jax.random.normal(rng1)
            if self.constantnoise_type == "white":
                _selected_muscle_control += self.constantnoise_level*_noise
            elif self.constantnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.constantnoise_level*_noise
            elif self.constantnoise_type == "red":
                self.constantnoise_acc += self.constantnoise_level*_noise
                _selected_muscle_control += self.constantnoise_acc
            else:
                raise NotImplementedError(f"{self.constantnoise_type}")

        # Update smoothed online estimate of motor actuation
        self._motor_act = (1 - self._motor_alpha) * state.info['motor_act'] \
                                + self._motor_alpha * jp.clip(_selected_motor_control, 0, 1)

        new_ctrl = new_ctrl.at[self._motor_actuators].set(self.sys.mj_model.actuator_ctrlrange[self._motor_actuators, 0] + self._motor_act*(self.sys.mj_model.actuator_ctrlrange[self._motor_actuators, 1] - self.sys.mj_model.actuator_ctrlrange[self._motor_actuators, 0]))
        new_ctrl = new_ctrl.at[self._muscle_actuators].set(jp.clip(_selected_muscle_control, 0, 1))

        isNormalized = False  #TODO: check whether we can integrate the default normalization from BaseV0.step
        
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


# class EpisodeWrapper(Wrapper):
#   """Maintains episode step count and sets done at episode end."""

#   def __init__(self, env: Env, episode_length: int, action_repeat: int):
#     super().__init__(env)
#     self.episode_length = episode_length
#     self.action_repeat = action_repeat

#   def reset(self, rng: jax.Array) -> State:
#     state = self.env.reset(rng)
#     state.info['steps'] = jp.zeros(rng.shape[:-1])
#     state.info['truncation'] = jp.zeros(rng.shape[:-1])
#     # Keep separate record of episode done as state.info['done'] can be erased
#     # by AutoResetWrapper
#     state.info['episode_done'] = jp.zeros(rng.shape[:-1])
#     episode_metrics = dict()
#     episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
#     episode_metrics['length'] = jp.zeros(rng.shape[:-1])
#     for metric_name in state.metrics.keys():
#       episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
#     state.info['episode_metrics'] = episode_metrics
#     return state

#   def step(self, state: State, action: jax.Array) -> State:
#     def f(state, _):
#       nstate = self.env.step(state, action)
#       return nstate, nstate.reward

#     state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
#     state = state.replace(reward=jp.sum(rewards, axis=0))
#     steps = state.info['steps'] + self.action_repeat
#     one = jp.ones_like(state.done)
#     zero = jp.zeros_like(state.done)
#     episode_length = jp.array(self.episode_length, dtype=jp.int32)
#     done = jp.where(steps >= episode_length, one, state.done)
#     state.info['truncation'] = jp.where(
#         steps >= episode_length, 1 - state.done, zero
#     )
#     state.info['steps'] = steps

#     # Aggregate state metrics into episode metrics
#     prev_done = state.info['episode_done']
#     state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
#     state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
#     state.info['episode_metrics']['length'] += self.action_repeat
#     state.info['episode_metrics']['length'] *= (1 - prev_done)
#     for metric_name in state.metrics.keys():
#       if metric_name != 'reward':
#         state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
#         state.info['episode_metrics'][metric_name] *= (1 - prev_done)
#     state.info['episode_done'] = done
#     return state.replace(done=done)
