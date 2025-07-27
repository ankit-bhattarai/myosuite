""" =================================================
# Copyright (c) User-in-the-Box 2024; Facebook, Inc. and its affiliates
Authors  :: Florian Fischer (fjf33@cam.ac.uk); Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import collections
import functools
import abc

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
from mujoco_playground._src import mjx_env
from ml_collections import config_dict


def get_default_config():
    return config_dict.create(
        model_path="myosuite/simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml",
        frame_skip=25,
        ctrl_dt=0.002 * 25,  # Each control step is 25 physics steps
        sim_dt=0.002,
        vision=False,
        num_envs=1024,
        obs_keys=[
            "qpos",
            "qvel",
            "qacc",
            "ee_pos",
            "act",
            "motor_act",
        ],
        weighted_reward_keys={
            "reach": 1.0,
            "bonus": 8.0,
            "neural_effort": 0,  # 1e-4,
        },
        target_pos_range={
            "fingertip": jp.array([[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]]),
        },
        target_radius_range={
            "fingertip": jp.array([0.05, 0.05]),
        },
        target_origin_rel=jp.zeros(3),
        ref_site="humphant",
        adaptive_params=config_dict.create(
            init_target_area_width_scale=1.0,
            adaptive_increase_success_rate=1.1,
            adaptive_decrease_success_rate=-0.1,
            adaptive_change_step_size=0.05,
            adaptive_change_min_trials=50,
            success_log_buffer_length=500,
        ),
        muscle_condition=None,
        sex=None,
        max_trials=10,
        noise_params=config_dict.create(
            sigdepnoise_type=None,
            sigdepnoise_level=0.103,
            constantnoise_type=None,
            constantnoise_level=0.185,
        ),
        reset_type="range_uniform",
        episode_length=800,
        distance_reach_metric_coefficient=10.0,
        x_reach_metric_coefficient=2.0,
        x_reach_weight=1.0,
        success_bonus=50.0,
        phase_0_to_1_transition_bonus=0.0,
    )


ALLOWED_VISION_MODES = ("rgbd", "depth", "depth_w_aux_task")


class MyoUserBase(mjx_env.MjxEnv):
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def action_size(self) -> int:
        return self._nu

    @property
    def xml_path(self) -> str:
        return self._config.model_path

    def prepare_mjx_model(self):
        spec = mujoco.MjSpec.from_file(self._config.model_path)

        mj_model = spec.compile()

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.timestep = self._config.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(self._mj_model)

    def _prepare_vision(self):
        if not self._config.vision:
            self.vision = False
            return
        self.vision = True
        from madrona_mjx.renderer import BatchRenderer

        self.vision_mode = self._config.vision.vision_mode
        assert (
            self.vision_mode in ALLOWED_VISION_MODES
        ), f"Invalid vision mode: {self.vision_mode} (allowed modes: {ALLOWED_VISION_MODES})"

        self.batch_renderer = BatchRenderer(
            m=self._mjx_model,
            gpu_id=self._config.vision.gpu_id,
            num_worlds=self._config.num_envs,
            batch_render_view_width=self._config.vision.render_width,
            batch_render_view_height=self._config.vision.render_height,
            enabled_geom_groups=np.asarray([0, 1, 2]),
            enabled_cameras=np.asarray(self._config.vision.enabled_cameras),
            use_rasterizer=False,
            viz_gpu_hdls=None,
            add_cam_debug_geo=False,
        )

    def __init__(
        self,
        config: config_dict.ConfigDict = get_default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        self.prepare_mjx_model()
        self._prepare_env()
        self._prepare_vision()

    def _prepare_env(self):
        self._prepare_bm_model()
        self._setup()

        # Do a forward step so stuff like geom and body positions are calculated
        # [using MjData rather than mjx.Data, to reduce computational overheat]
        _data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, _data)

        self._prepare_after_init(_data)

    @abc.abstractmethod
    def _setup(self):
        """Task specific setup"""

    @abc.abstractmethod
    def _prepare_after_init(self, data):
        """Task specific after init"""

    def _prepare_bm_model(self):
        # Total number of actuators
        self._nu = self._mjx_model.nu

        # Number of muscle actuators
        self._na = self._mjx_model.na

        # Number of motor actuators
        self._nm = self._nu - self._na
        self._motor_act = jp.zeros((self._nm,))
        self._motor_alpha = 0.9 * jp.ones(1)

        # Get actuator names (muscle and motor)
        self._actuator_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self._mjx_model.nu)
        ]
        self._muscle_actuator_names = set(
            np.array(self._actuator_names)[
                self._mjx_model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
            ]
        )  # model.actuator_dyntype==mujoco.mjtDyn.mjDYN_MUSCLE
        self._motor_actuator_names = (
            set(self._actuator_names) - self._muscle_actuator_names
        )

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._muscle_actuator_names = sorted(
            self._muscle_actuator_names, key=self._actuator_names.index
        )
        self._motor_actuator_names = sorted(
            self._motor_actuator_names, key=self._actuator_names.index
        )

        # Find actuator indices in the simulation
        self._muscle_actuators = jp.array(
            [
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                for actuator_name in self._muscle_actuator_names
            ],
            dtype=jp.int32,
        )
        self._motor_actuators = jp.array(
            [
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                for actuator_name in self._motor_actuator_names
            ],
            dtype=jp.int32,
        )

        # Get joint names (dependent and independent)
        self._joint_names = [
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self._mjx_model.njnt)
        ]
        self._dependent_joint_names = (
            {
                self._joint_names[idx]
                for idx in np.unique(
                    self._mjx_model.eq_obj1id[self._mjx_model.eq_active0.astype(bool)]
                )
            }
            if self._mjx_model.eq_obj1id is not None
            else set()
        )
        self._independent_joint_names = (
            set(self._joint_names) - self._dependent_joint_names
        )

        # Sort the names to preserve original ordering (not really necessary but looks nicer)
        self._dependent_joint_names = sorted(
            self._dependent_joint_names, key=self._joint_names.index
        )
        self._independent_joint_names = sorted(
            self._independent_joint_names, key=self._joint_names.index
        )

        # Find dependent and independent joint indices in the simulation
        self._dependent_joints = [
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in self._dependent_joint_names
        ]
        self._independent_joints = [
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in self._independent_joint_names
        ]

        # If there are 'free' type of joints, we'll need to be more careful with which dof corresponds to
        # which joint, for both qpos and qvel/qacc. There should be exactly one dof per independent/dependent joint.
        def get_dofs(joint_indices):
            qpos = jp.array([], dtype=jp.int32)
            dofs = jp.array([], dtype=jp.int32)
            for joint_idx in joint_indices:
                if self._mjx_model.jnt_type[joint_idx] not in [
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ]:
                    raise NotImplementedError(
                        f"Only 'hinge' and 'slide' joints are supported, joint "
                        f"{self._joint_names[joint_idx]} is of type {mujoco.mjtJoint(self._mjx_model.jnt_type[joint_idx]).name}"
                    )
                qpos = jp.append(qpos, self._mjx_model.jnt_qposadr[joint_idx])
                dofs = jp.append(dofs, self._mjx_model.jnt_dofadr[joint_idx])
            return qpos, dofs

        self._dependent_qpos, self._dependent_dofs = get_dofs(self._dependent_joints)
        self._independent_qpos, self._independent_dofs = get_dofs(
            self._independent_joints
        )

    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        self.muscle_condition = self._config.muscle_condition
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self._mj_model.actuator_gainprm.shape[0]):
                self._mj_model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self._mj_model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self._mj_model,
                frame_skip=self.frame_skip,
                sex=self.sex,
                seed=self.get_input_seed(),
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self._mj_model.actuator_name2id("EPL")
            self.EIPpos = self._mj_model.actuator_name2id("EIP")

    def ensure_dependent_joint_angles(self, data):
        """Adjusts virtual joints according to active joint constraints."""

        _joint_constraints = jp.array(self._mj_model.eq_type == 2)
        _active_eq_constraints = data.eq_active == 1
        indices = _joint_constraints & _active_eq_constraints
        poly_coefs_rows = jp.take(self._mj_model.eq_data, indices, axis=0)
        for virtual_joint_id, physical_joint_id, poly_coefs in zip(
            jp.take(self._mj_model.eq_obj1id, indices),
            jp.take(self._mj_model.eq_obj2id, indices),
            jp.take(poly_coefs_rows, jp.array([4, 3, 2, 1, 0]), axis=1),
        ):
            old_qpos = data.qpos
            new_qpos = data.qpos.copy()
            new_qpos = new_qpos.at[virtual_joint_id].set(
                jp.polyval(poly_coefs, data.qpos[physical_joint_id])
            )
            data = data.replace(
                qpos=jp.select([physical_joint_id >= 0], [new_qpos], old_qpos)
            )
        return data

    def _reset_bm_model(self, rng):
        # TODO: do not store anything in self in this function, as its values should mostly be discarded after it is called (no permanent env changes!)

        # Sample random initial values for motor activation
        rng, rng1 = jax.random.split(rng, 2)
        self._motor_act = jax.random.uniform(
            rng1,
            shape=(self._nm,),
            minval=jp.zeros((self._nm,)),
            maxval=jp.ones((self._nm,)),
        )
        # Reset smoothed average of motor actuator activation
        self._motor_smooth_avg = jp.zeros((self._nm,))

        # Reset accumulative noise
        zero = jp.zeros(1)
        self._sigdepnoise_acc = zero
        self._constantnoise_acc = zero

    def _reset_zero(self, rng):
        """Resets the biomechanical model."""

        # Set joint angles and velocities to zero
        rng, rng1 = jax.random.split(rng, 2)
        nqi = len(self._independent_qpos)
        qpos = jp.zeros((nqi,))
        qvel = jp.zeros((nqi,))
        reset_qpos = jp.zeros((self._mj_model.nq,))
        reset_qvel = jp.zeros((self._mj_model.nv,))

        # Randomly sample act within unit interval
        reset_act = jax.random.uniform(
            rng1,
            shape=self._na,
            minval=jp.zeros((self._na,)),
            maxval=jp.ones((self._na,)),
        )

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
        """Resets the biomechanical model."""

        # Randomly sample qpos and qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        qpos = jax.random.uniform(
            rng1,
            shape=nqi,
            minval=jp.ones((nqi,)) * -0.05,
            maxval=jp.ones((nqi,)) * 0.05,
        )
        qvel = jax.random.uniform(
            rng2,
            shape=nqi,
            minval=jp.ones((nqi,)) * -0.05,
            maxval=jp.ones((nqi,)) * 0.05,
        )
        reset_qpos = jp.zeros((self._mj_model.nq,))
        reset_qvel = jp.zeros((self._mj_model.nv,))
        reset_act = jax.random.uniform(
            rng3,
            shape=self._na,
            minval=jp.zeros((self._na,)),
            maxval=jp.ones((self._na,)),
        )

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
        """Resets the biomechanical model."""

        # Randomly sample qpos within joint range, qvel around zero values, and act within unit interval
        nqi = len(self._independent_qpos)
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        jnt_range = self._mj_model.jnt_range[self._independent_joints]
        qpos = jax.random.uniform(
            rng1, shape=(nqi,), minval=jnt_range[:, 0], maxval=jnt_range[:, 1]
        )
        qvel = jax.random.uniform(
            rng2,
            shape=(nqi,),
            minval=jp.ones((nqi,)) * -0.05,
            maxval=jp.ones((nqi,)) * 0.05,
        )
        reset_qpos = jp.zeros((self._mj_model.nq,))
        reset_qvel = jp.zeros((self._mj_model.nv,))
        reset_act = jax.random.uniform(
            rng3,
            shape=self._na,
            minval=jp.zeros((self._na,)),
            maxval=jp.ones((self._na,)),
        )

        # Set qpos and qvel
        reset_qpos = reset_qpos.at[self._independent_qpos].set(qpos)
        # reset_qpos[self._dependent_qpos] = 0
        reset_qvel = reset_qvel.at[self._independent_dofs].set(qvel)
        # reset_qvel[self._dependent_dofs] = 0
        data = self.ensure_dependent_joint_angles(data)

        # # Randomly sample act within unit interval
        # act = self.np_random.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))
        # self.sim.data.act[self._muscle_actuators] = act

        return reset_qpos, reset_qvel, reset_act

    def generate_pixels(self, data, render_token=None):
        # Generates the view of the environment using the batch renderer
        update_info = {}
        if render_token is None:  # Initialize renderer during reset
            render_token, rgb, depth = self.batch_renderer.init(data, self._mjx_model)
            update_info.update({"render_token": render_token})
        else:  # Render during step
            _, rgb, depth = self.batch_renderer.render(render_token, data)
        pixels = rgb[0][..., :3].astype(jp.float32) / 255.0
        depth = depth[0].astype(jp.float32)

        if self.vision_mode == "rgb":
            update_info.update({"pixels/view_0": pixels})
        elif self.vision_mode == "rgbd" or self.vision_mode == "rgbd_only":
            # combine pixels and depth into a single image
            rgbd = jp.concatenate([pixels, depth], axis=-1)
            update_info.update({"pixels/view_0": rgbd})
        elif (
            self.vision_mode == "rgb+depth"
            or self.vision_mode == "depth_only"
            or self.vision_mode == "depth"
            or self.vision_mode == "depth_w_aux_task"
        ):
            update_info.update({"pixels/view_0": pixels, "pixels/depth": depth})
        else:
            raise ValueError(f"Invalid vision mode: {self.vision_mode}")
        return update_info