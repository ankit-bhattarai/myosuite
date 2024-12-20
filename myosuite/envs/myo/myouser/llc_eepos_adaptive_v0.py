""" =================================================
# Copyright (c) User-in-the-Box 2024; Facebook, Inc. and its affiliates
Authors  :: Florian Fischer (fjf33@cam.ac.uk); Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import mujoco
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0



class LLCEEPosAdaptiveEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'qacc', 'act', 'motor_act', 'ee_pos', 'target_pos', 'target_radius']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 8.0,
        #"penalty": 50,
        "neural_effort": 1e-4,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=None)

        self._prepare_env(self.sim.model, **kwargs)
    
    def _prepare_env(self, model, **kwargs):
        self._prepare_bm_model(model)
        self._prepare_reaching_task_pt1(model)
        self._setup(**kwargs)
        self._prepare_reaching_task_pt2(model)

    def _prepare_bm_model(self, model):
        # Total number of actuators
        self._nu = model.nu

        # Number of muscle actuators
        self._na = model.na

        # Number of motor actuators
        self._nm = self._nu - self._na
        self._motor_act = np.zeros((self._nm,))
        self._motor_alpha = 0.9

        # Get actuator names (muscle and motor)
        self._actuator_names = [mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        self._muscle_actuator_names = set(np.array(self._actuator_names)[model.actuator_trntype==mujoco.mjtTrn.mjTRN_TENDON])  #model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE
        self._motor_actuator_names = set(self._actuator_names) - self._muscle_actuator_names

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._muscle_actuator_names = sorted(self._muscle_actuator_names, key=self._actuator_names.index)
        self._motor_actuator_names = sorted(self._motor_actuator_names, key=self._actuator_names.index)

        # Find actuator indices in the simulation
        self._muscle_actuators = [mujoco.mj_name2id(model._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                                for actuator_name in self._muscle_actuator_names]
        self._motor_actuators = [mujoco.mj_name2id(model._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                                for actuator_name in self._motor_actuator_names]

        # Get joint names (dependent and independent)
        self._joint_names = [mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        self._dependent_joint_names = {self._joint_names[idx] for idx in
                                    np.unique(model.eq_obj1id[model.eq_active0.astype(bool)])} \
        if model.eq_obj1id is not None else set()
        self._independent_joint_names = set(self._joint_names) - self._dependent_joint_names

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._dependent_joint_names = sorted(self._dependent_joint_names, key=self._joint_names.index)
        self._independent_joint_names = sorted(self._independent_joint_names, key=self._joint_names.index)

        # Find dependent and independent joint indices in the simulation
        self._dependent_joints = [mujoco.mj_name2id(model._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                for joint_name in self._dependent_joint_names]
        self._independent_joints = [mujoco.mj_name2id(model._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                for joint_name in self._independent_joint_names]

        # If there are 'free' type of joints, we'll need to be more careful with which dof corresponds to
        # which joint, for both qpos and qvel/qacc. There should be exactly one dof per independent/dependent joint.
        def get_dofs(joint_indices):
            qpos = []
            dofs = []
            for joint_idx in joint_indices:
                if model.jnt_type[joint_idx] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                    raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                            f"{self._joint_names[joint_idx]} is of type {mujoco.mjtJoint(model.jnt_type[joint_idx]).name}")
                qpos.append(model.jnt_qposadr[joint_idx])
                dofs.append(model.jnt_dofadr[joint_idx])
            return qpos, dofs
        self._dependent_qpos, self._dependent_dofs = get_dofs(self._dependent_joints)
        self._independent_qpos, self._independent_dofs = get_dofs(self._independent_joints)

    def _prepare_reaching_task_pt1(self, model):
    	# Internal variables
        self._trial_idx = 0  #number of total trials since last reset
        self._targets_hit = 0       
        self._trial_success_log = []  #logging whether trials since last target limit adjustment were successful
        self.n_hits_adj = 0  #number of successful trials since last target limit adjustment
        self.n_targets_adj = 0  #number of total trials since last target limit adjustment
        self.n_adjs = 0  #number of target limit adjustments
        self.success_rate = 0  #previous success rate

    def _prepare_reaching_task_pt2(self, model):
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = self.sim.data.site_xpos[self.sim.model.site_name2id(self.ref_site)] if self.ref_site is not None else np.zeros(3,)
        
        # Dwelling based selection -- fingertip needs to be inside target for some time
        self.dwell_threshold = 0.25/self.dt  #corresponds to 250ms; for visual-based pointing use 0.5/self.dt
        
        # Use early termination if target is not hit in time
        self._steps_since_last_hit = 0
        self._max_steps_without_hit = 4./self.dt #corresponds to 4 seconds
    
    def _setup(self,
            target_pos_range:dict,
            target_radius_range:dict,
            ref_site = None,
            adaptive_task = False,
            init_target_area_width_scale = 0,
            adaptive_increase_success_rate = 0.6,
            adaptive_decrease_success_rate = 0.3,
            adaptive_change_step_size = 0.05,
            adaptive_change_min_trials = 50,
            adaptive_change_trial_buffer_length = None,
            frame_skip = 25,  #10,
            max_trials = 10,
            sigdepnoise_type = None,   #"white"
            sigdepnoise_level = 0.103,
            constantnoise_type = None,   #"white"
            constantnoise_level = 0.185,
            reset_type = "range_uniform",
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.target_pos_range = target_pos_range
        self.target_radius_range = target_radius_range
        self.ref_site = ref_site

        # Define a maximum number of trials per episode (if needed for e.g. evaluation / visualisation)
        self.max_trials = max_trials

        self.adaptive_task = adaptive_task
        if self.adaptive_task:
            # Additional variables needed for adaptive adjustment of target limits based on success rates since last target limit adjustment
            self.target_area_dynamic_width_scale = init_target_area_width_scale  #scale factor for target area width (between 0 and 1), i.e., the percentage of the target limit range defined above that is currently used when spawning targets
            self.adaptive_increase_success_rate = adaptive_increase_success_rate  #success rate above which target area width is increased
            self.adaptive_decrease_success_rate = adaptive_decrease_success_rate  #success rate below which target area width is decreased
            self.adaptive_change_step_size = adaptive_change_step_size  #increase of target area width per adjustment (in meter)
            self.adaptive_change_min_trials = adaptive_change_min_trials  #minimum number of trials with the latest target area width required before the next adjustment; should be chosen considerably larger than self.max_trials
            self.adaptive_change_trial_buffer_length = adaptive_change_trial_buffer_length  #maximum number of trials (since last adjustment) to consider for success rate calculation (default: consider all values since last adjustment)
            assert self.adaptive_change_min_trials >= 1, f"At least one trial is required to assess the success rate for adaptively adjusting the target area width. Set 'adaptive_change_min_trials' >= 1 (current value: {self.adaptive_change_min_trials})."
        
        # Define signal-dependent noise
        self.sigdepnoise_type = sigdepnoise_type 
        self.sigdepnoise_level = sigdepnoise_level
        self.sigdepnoise_acc = 0  #only used for red/Brownian noise

        # Define constant (i.e., signal-independent) noise
        self.constantnoise_type = constantnoise_type
        self.constantnoise_level = constantnoise_level
        self.constantnoise_acc = 0  #only used for red/Brownian noise

        # Define reset type
        self.reset_type = reset_type
        ## valid reset types: 
        valid_reset_types = ("zero", "epsilon_uniform", "range_uniform", None)
        assert self.reset_type in valid_reset_types, f"Invalid reset type '{self.reset_type} (valid types are {valid_reset_types})."

        # Initialise other variables required for setup (i.e., by get_obs_vec, get_obs_dict, get_reward_dict, or get_env_infos)
        self.dwell_threshold = 0.
        self.target_coordinates_origin = self.sim.data.site_xpos[self.sim.model.site_name2id(self.ref_site)] if self.ref_site is not None else np.zeros(3,)

        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_pos_range.keys(),
                frame_skip=frame_skip,
                **kwargs,
                )
    
    # step the simulation forward (overrides BaseV0.step; --> use control smoothening instead of normalisation; also, enable signal-dependent and/or constant motor noise)
    def step(self, a, **kwargs):
        new_ctrl = a.copy()

        _selected_motor_control = np.clip(self._motor_act + a[:self._nm], 0, 1)
        _selected_muscle_control = np.clip(self.sim.data.act[self._muscle_actuators] + a[self._nm:], 0, 1)

        if self.sigdepnoise_type is not None:
            if self.sigdepnoise_type == "white":
                _added_noise = self.sigdepnoise_level*self.np_random.normal(scale=_selected_muscle_control)
                _selected_muscle_control += _added_noise
            elif self.sigdepnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.sigdepnoise_level*self.np_random.normal(scale=_selected_muscle_control)
            elif self.sigdepnoise_type == "red":
                # self.sigdepnoise_acc *= 1 - 0.1
                self.sigdepnoise_acc += self.sigdepnoise_level*self.np_random.normal(scale=_selected_muscle_control)
                _selected_muscle_control += self.sigdepnoise_acc
            else:
                raise NotImplementedError(f"{self.sigdepnoise_type}")
        if self.constantnoise_type is not None:
            if self.constantnoise_type == "white":
                _selected_muscle_control += self.constantnoise_level*self.np_random.normal(scale=1)
            elif self.constantnoise_type == "whiteonly":  #only for debugging purposes
                _selected_muscle_control = self.constantnoise_level*self.np_random.normal(scale=1)
            elif self.constantnoise_type == "red":
                self.constantnoise_acc += self.constantnoise_level*self.np_random.normal(scale=1)
                _selected_muscle_control += self.constantnoise_acc
            else:
                raise NotImplementedError(f"{self.constantnoise_type}")

        # Update smoothed online estimate of motor actuation
        self._motor_act = (1 - self._motor_alpha) * self._motor_act \
                                + self._motor_alpha * np.clip(_selected_motor_control, 0, 1)

        new_ctrl[self._motor_actuators] = self.sim.model.actuator_ctrlrange[self._motor_actuators, 0] + self._motor_act*(self.sim.model.actuator_ctrlrange[self._motor_actuators, 1] - self.sim.model.actuator_ctrlrange[self._motor_actuators, 0])
        new_ctrl[self._muscle_actuators] = np.clip(_selected_muscle_control, 0, 1)

        isNormalized = False  #TODO: check whether we can integrate the default normalization from BaseV0.step
        

        ##### rest is re-implemented from BaseV0.step

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            new_ctrl[self._muscle_actuators], _, _ = self.muscle_fatigue.compute_act(
                new_ctrl[self._muscle_actuators]
            )
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            new_ctrl[self.EPLpos] = new_ctrl[self.EIPpos].copy()
            # Set EIP to 0
            new_ctrl[self.EIPpos] = 0
        
        # step forward
        self.last_ctrl = self.robot.step(
            ctrl_desired=new_ctrl,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )

        return self.forward(**kwargs)
    
    # # updates executed at each step, after MuJoCo step (see BaseV0.step) but before MyoSuite returns observations, reward and infos (see MujocoEnv.forward)
    # def _forward(self, **kwargs):
    #     pass
        
    #     # continue with default forward step
    #     super()._forward(**kwargs)

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])

        # Normalise qpos
        jnt_range = self.sim.model.jnt_range[self._independent_joints]
        qpos = self.sim.data.qpos[self._independent_qpos].copy()
        qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
        qpos = (qpos - 0.5) * 2
        self.obs_dict['qpos'] = qpos

        # Get qvel, qacc
        self.obs_dict['qvel'] = self.sim.data.qvel[self._independent_dofs].copy()  #*self.dt
        self.obs_dict['qacc'] = self.sim.data.qacc[self._independent_dofs].copy()

        # Normalise act
        if self.sim.model.na>0:
            self.obs_dict['act']  = (self.sim.data.act.copy() - 0.5) * 2

        # Smoothed average of motor actuation (only for motor actuators); normalise
        self.obs_dict['motor_act'] = (self._motor_act.copy() - 0.5) * 2

        # End-effector and target position
        self.obs_dict['ee_pos'] = np.row_stack([self.sim.data.site_xpos[self.tip_sids[isite]].copy() for isite in range(len(self.tip_sids))])
        self.obs_dict['target_pos'] = np.row_stack([self.sim.data.site_xpos[self.target_sids[isite]].copy() for isite in range(len(self.tip_sids))])

        # Distance to target (used for rewards later)
        self.obs_dict['reach_dist'] = np.linalg.norm(np.array(self.obs_dict['target_pos']) - np.array(self.obs_dict['ee_pos']), axis=-1)

        # Target radius
        self.obs_dict['target_radius'] = np.array([self.sim.model.site_size[self.target_sids[isite]][0] for isite in range(len(self.tip_sids))])

        # Task progress/success metrics
        if np.all(self.obs_dict['reach_dist'] < self.obs_dict['target_radius']):  ## we require all end-effector--target pairs to have distance below the respective target radius
            self._steps_inside_target += 1
        else:
            self._steps_inside_target = 0
        self.obs_dict['steps_inside_target'] = np.array([self._steps_inside_target])
        self.obs_dict['target_hit'] = np.array([self.obs_dict['steps_inside_target'] >= self.dwell_threshold])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        
        # Normalise qpos
        jnt_range = sim.model.jnt_range[self._independent_joints]
        qpos = sim.data.qpos[self._independent_qpos].copy()
        qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
        qpos = (qpos - 0.5) * 2
        obs_dict['qpos'] = qpos

        # Get qvel, qacc
        obs_dict['qvel'] = sim.data.qvel[self._independent_dofs].copy()  #*self.dt
        obs_dict['qacc'] = sim.data.qacc[self._independent_dofs].copy()

        # Normalise act
        if sim.model.na>0:
            obs_dict['act']  = (sim.data.act.copy() - 0.5) * 2
        obs_dict['last_ctrl'] = self.last_ctrl

        # Smoothed average of motor actuation (only for motor actuators); normalise
        obs_dict['motor_act'] = (self._motor_act.copy() - 0.5) * 2

        # End-effector and target position
        obs_dict['ee_pos'] = np.row_stack([sim.data.site_xpos[self.tip_sids[isite]].copy() for isite in range(len(self.tip_sids))])
        obs_dict['target_pos'] = np.row_stack([sim.data.site_xpos[self.target_sids[isite]].copy() for isite in range(len(self.tip_sids))])

        # Distance to target (used for rewards later)
        obs_dict['reach_dist'] = np.linalg.norm(np.array(obs_dict['target_pos']) - np.array(obs_dict['ee_pos']), axis=-1)

        # Target radius
        obs_dict['target_radius'] = np.array([sim.model.site_size[self.target_sids[isite]][0] for isite in range(len(self.tip_sids))])

        # Task progress/success metrics
        if np.all(obs_dict['reach_dist'] < obs_dict['target_radius']):  ## we require all end-effector--target pairs to have distance below the respective target radius
            self._steps_inside_target += 1
        else:
            self._steps_inside_target = 0
        obs_dict['steps_inside_target'] = np.array([self._steps_inside_target])
        obs_dict['target_hit'] = np.array([obs_dict['steps_inside_target'] >= self.dwell_threshold])
        obs_dict['trial_idx'] = np.array([self._trial_idx])

        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = obs_dict['reach_dist']
        target_radius = obs_dict['target_radius']
        reach_dist_to_target_bound = np.linalg.norm(np.moveaxis(reach_dist-target_radius, -1, -2), axis=-1)
        steps_inside_target = np.linalg.norm(obs_dict['steps_inside_target'], axis=-1)
        last_ctrl = np.linalg.norm(obs_dict['last_ctrl'], axis=-1)
        trial_idx = np.linalg.norm(obs_dict['trial_idx'], axis=-1)

        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        # far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        # near_th = len(self.tip_sids)*.0125
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   (np.exp(-reach_dist_to_target_bound*10.) - 1.)/10.),  #-1.*reach_dist),
            ('bonus',   1.*(steps_inside_target >= self.dwell_threshold)),  #1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('neural_effort', -1.*(last_ctrl ** 2)),
            ('act_reg', -1.*act_mag),
            # ('penalty', -1.*(np.any(reach_dist > far_th))),
            # Must keys
            ('sparse',  -1.*(np.linalg.norm(reach_dist, axis=-1) ** 2)),
            ('solved',  steps_inside_target >= self.dwell_threshold),
            ('done',    trial_idx >= self.max_trials), #np.any(reach_dist > far_th))),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def get_env_infos(self):
        """
        Get information about the environment.
        """
        env_info = super().get_env_infos()

        if self.obs_dict['target_hit']:
            self._trial_idx += 1
            self._targets_hit += 1
            self._trial_success_log += [1]
            self._steps_since_last_hit = 0
            self._steps_inside_target = 0
            self.generate_target()
        else:
            self._steps_since_last_hit += 1
            
            if self._steps_since_last_hit >= self._max_steps_without_hit:
                self._trial_success_log += [0]
                # Spawn a new target
                self._steps_since_last_hit = 0
                self._trial_idx += 1
                self.generate_target()
        
        env_info_additional = {
            'target_area_dynamic_width_scale': self.target_area_dynamic_width_scale,
            'success_rate': self.success_rate,
        }

        env_info.update(env_info_additional)
        return env_info

    # generate a valid target
    def generate_target(self, new_positions=None, new_radii=None):        
        # Check if target area width should be updated
        self.check_adaptive_target_area_width()

        # Set target location
        if new_positions is None:
            # Sample target position
            for site, span in self.target_pos_range.items():
                if self.adaptive_task:
                    span = self.get_current_target_pos_range(span)
                sid = self.sim.model.site_name2id(site+'_target')
                new_position = self.target_coordinates_origin + self.np_random.uniform(low=span[0], high=span[1])
                self.sim.model.site_pos[sid] = new_position
        
        # Set target size
        if new_radii is None:
            # Sample target radius
            for site, span in self.target_radius_range.items():
                sid = self.sim.model.site_name2id(site+'_target')
                new_radius = self.np_random.uniform(low=span[0], high=span[1])
                self.sim.model.site_size[sid][0] = new_radius

        self.sim.forward()

    def get_current_target_pos_range(self, span):
        return self.target_area_dynamic_width_scale*(span - np.mean(span, axis=0)) + np.mean(span, axis=0)
    
    def check_adaptive_target_area_width(self):
        if self.adaptive_change_trial_buffer_length is not None:
            # Clip success/fail buffer
            self._trial_success_log = self._trial_success_log[-self.adaptive_change_trial_buffer_length:]

        self.n_targets_adj = len(self._trial_success_log)
        if self.n_targets_adj >= self.adaptive_change_min_trials:
        
            self.n_hits_adj = sum(self._trial_success_log)
            self.success_rate = self.n_hits_adj / self.n_targets_adj
            # print(f"SUCCESS RATE: {self.success_rate*100}% ({self.n_hits_adj}/{self.n_targets_adj}) -- Last Adj. #{self.n_adjs}")

            if (self.success_rate >= self.adaptive_increase_success_rate) and (self.target_area_dynamic_width_scale < 1):
                new_target_area_width = self.target_area_dynamic_width_scale + self.adaptive_change_step_size
                self.update_adaptive_target_area_width(new_target_area_width)
            elif (self.success_rate <= self.adaptive_decrease_success_rate) and (self.target_area_dynamic_width_scale > 0):
                new_target_area_width = self.target_area_dynamic_width_scale - self.adaptive_change_step_size
                self.update_adaptive_target_area_width(new_target_area_width)

    def update_adaptive_target_area_width(self, new_target_area_width):
        self.n_adjs += 1
        print(f"ADAPTIVE TARGETS -- Adj. #{self.n_adjs}: {self.target_area_dynamic_width_scale*100}% -> {new_target_area_width*100}% (success_rate={self.success_rate})")

        # Reset internally used counters
        self._trial_success_log = []
        self.n_hits_adj = 0  #TODO: remove (useless)
        self.n_targets_adj = 0  #TODO: remove (useless)

        self.target_area_dynamic_width_scale = new_target_area_width

    def reset(self, **kwargs):
        # Reset counters
        self._steps_since_last_hit = 0
        self._steps_inside_target = 0
        self._trial_idx = 0
        self._targets_hit = 0

        self.generate_target()
        self.robot.sync_sims(self.sim, self.sim_obsd)

        if self.reset_type == "zero":
            reset_qpos, reset_qvel = self._reset_zero()
        elif self.reset_type == "epsilon_uniform":
            reset_qpos, reset_qvel = self._reset_epsilon_uniform()
        elif self.reset_type == "range_uniform":
            reset_qpos, reset_qvel = self._reset_range_uniform()
        else:
            reset_qpos, reset_qvel = None, None
        
        obs = super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs)

        self._reset_bm_model()
        return obs
    
    def _reset_zero(self):
        """ Resets the biomechanical model. """

        # Set joint angles and velocities to zero
        nqi = len(self._independent_qpos)
        qpos = np.zeros((nqi,))
        qvel = np.zeros((nqi,))
        reset_qpos = np.zeros((self.sim.model.nq,))
        reset_qvel = np.zeros((self.sim.model.nv,))

        # Set qpos and qvel
        reset_qpos[self._dependent_qpos] = 0
        reset_qpos[self._independent_qpos] = qpos
        reset_qvel[self._dependent_dofs] = 0
        reset_qvel[self._independent_dofs] = qvel

        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel
    
    def _reset_epsilon_uniform(self):
        """ Resets the biomechanical model. """

        # Randomly sample qpos and qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        qpos = self.np_random.uniform(low=np.ones((nqi,))*-0.05, high=np.ones((nqi,))*0.05)
        qvel = self.np_random.uniform(low=np.ones((nqi,))*-0.05, high=np.ones((nqi,))*0.05)
        reset_qpos = np.zeros((self.sim.model.nq,))
        reset_qvel = np.zeros((self.sim.model.nv,))

        # Set qpos and qvel
        ## TODO: ensure that constraints are initially satisfied
        reset_qpos[self._dependent_qpos] = 0
        reset_qpos[self._independent_qpos] = qpos
        reset_qvel[self._dependent_dofs] = 0
        reset_qvel[self._independent_dofs] = qvel
        
        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel

    def _reset_range_uniform(self):
        """ Resets the biomechanical model. """

        # Randomly sample qpos within joint range, qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        jnt_range = self.sim.model.jnt_range[self._independent_joints]
        qpos = self.np_random.uniform(low=jnt_range[:, 0], high=jnt_range[:, 1])
        qvel = self.np_random.uniform(low=np.ones((nqi,))*-0.05, high=np.ones((nqi,))*0.05)
        reset_qpos = np.zeros((self.sim.model.nq,))
        reset_qvel = np.zeros((self.sim.model.nv,))

        # Set qpos and qvel
        reset_qpos[self._independent_qpos] = qpos
        # reset_qpos[self._dependent_qpos] = 0
        reset_qvel[self._independent_dofs] = qvel
        # reset_qvel[self._dependent_dofs] = 0
        self.ensure_dependent_joint_angles()
        
        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel

    def ensure_dependent_joint_angles(self):
        """ Adjusts virtual joints according to active joint constraints. """

        for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
                self.sim.model.eq_obj1id[
                    (self.sim.model.eq_type == 2) & (self.sim.data.eq_active == 1)],
                self.sim.model.eq_obj2id[
                    (self.sim.model.eq_type == 2) & (self.sim.data.eq_active == 1)],
                self.sim.model.eq_data[(self.sim.model.eq_type == 2) &
                                            (self.sim.data.eq_active == 1), 4::-1]):
            if physical_joint_id >= 0:
                self.sim.data.joint(virtual_joint_id).qpos = np.polyval(poly_coefs, self.sim.data.joint(physical_joint_id).qpos)

    def _reset_bm_model(self):
        # Sample random initial values for motor activation
        self._motor_act = self.np_random.uniform(low=np.zeros((self._nm,)), high=np.ones((self._nm,)))
        # Reset smoothed average of motor actuator activation
        self._motor_smooth_avg = np.zeros((self._nm,))

        # Reset accumulative noise
        self._sigdepnoise_acc = 0
        self._constantnoise_acc = 0