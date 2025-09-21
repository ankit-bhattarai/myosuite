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
from typing import List, Dict, Union, Any

@dataclass
class ReachSettings:
    ref_site: str = "humphant"
    target_origin_rel: List[float] = field(default_factory=lambda: [0., 0., 0.])


@dataclass
class PointingTarget:
    # penetrable: bool = False
    # Position can either be a 3d vector or a 2 x list of 3d vectors specifying the min and max values for each dimension
    position: List[List[float]] = field(
        default_factory=lambda: [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]])
    shape: str = "sphere"
    # Size can either be a single value or a list of 2 values specifying the min and max values
    size: List[float] = field(default_factory=lambda: [0.05, 0.15])
    # Any rewards received when inside the target
    reward_incentive: float = 0.0
    completion_bonus: float = 0.0
    dwell_duration: float = 0.25
    rgb: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

@dataclass
class UniversalTaskConfig:
    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: List[str] = field(default_factory=lambda: [
        "qpos",
        "qvel",
        "qacc",
        "ee_pos",
        "act",
    ])
    omni_keys: List[str] = field(default_factory=lambda: ["target_pos", "target_size", "phase"])
    weighted_reward_keys: Dict[str, float] = field(default_factory=lambda: {
        "reach": 1,
        "bonus": 8,
    })
    reach_metric: float = 10.0
    max_duration: float = 4.
    dwell_duration: float = 0.25
    max_trials: int = 1
    reset_type: str = "range_uniform"
    num_targets: int = 1
    targets: List[PointingTarget] = field(default_factory=lambda: [
        PointingTarget(completion_bonus=0.0),
        PointingTarget(completion_bonus=1.0, rgb=[0.0, 1.0, 0.0]),
    ])

@dataclass
class UniversalEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserUniversal"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_universal_myouser.xml"
    task_config: UniversalTaskConfig = field(default_factory=lambda: UniversalTaskConfig())

class PointingTargetClass:
    def __init__(self, phase_number: int, target_pos_range: List[List[float]], target_radius_range: List[float], 
                target_name: str = "ee_pos", shape: str = "sphere"):
        self.phase_number = phase_number
        self.target_coordinates_origin = jp.zeros(3) # TODO: to be set later
        self.target_pos_range = jp.array(target_pos_range)
        self.target_radius_range = jp.array(target_radius_range)
        self.target_name = target_name
        assert shape == "sphere", "Only sphere shapes are supported for now"
        self.shape = shape
        self.target_geom_id = None
        self.target_body_id = None

    def generate_target_pos(self, rng: jax.random.PRNGKey):
        target_pos = jax.random.uniform(rng, (3,), minval=self.target_pos_range[0], maxval=self.target_pos_range[1])
        target_pos = self.target_coordinates_origin + target_pos
        return target_pos

    def generate_target_size(self, rng: jax.random.PRNGKey):
        target_size = jax.random.uniform(rng, (3,), minval=self.target_radius_range[0], maxval=self.target_radius_range[1])
        return target_size

    def generate_target(self, rng: jax.random.PRNGKey):
        rng, pos_rng, size_rng = jax.random.split(rng, 3)
        target_pos = self.generate_target_pos(pos_rng)
        target_size = self.generate_target_size(size_rng)
        return target_pos, target_size

    @property
    def target_geom_name(self):
        return f"geom_target_{self.phase_number}"

    @property
    def target_body_name(self):
        return f"body_target_{self.phase_number}"
    
    def add_to_spec(self, spec: mujoco.MjSpec, rng: jax.random.PRNGKey, rgb: List[float]):
        worldbody = spec.worldbody
        target_pos, target_size = self.generate_target(rng)
        target_body = worldbody.add_body(name=self.target_body_name, pos=target_pos)
        rgba = jp.zeros(4)
        rgba = rgba.at[:3].set(rgb)
        target_geom = target_body.add_geom(name=self.target_geom_name, pos=jp.zeros(3), size=target_size, rgba=rgba)
        print(f"Added target {self.target_geom_name} to spec")
        return spec

    def reset(self, info: Dict[str, Any]):
        rng, rng_task = jax.random.split(info['rng'], 2)
        target_pos, target_size = self.generate_target(rng_task)
        info[f'phase_{self.phase_number}/target_pos'] = target_pos
        info[f'phase_{self.phase_number}/target_size'] = target_size
        info['rng'] = rng
        return info
    
    def target_step(self, state: State, action: jp.ndarray, info: Dict[str, Any]):
        pass

    def update_task_visuals(self, mj_model: mujoco.MjModel, state: State, phase: int):
        if phase != self.phase_number:
            mj_model.geom(self.target_geom_name).rgba[-1] = 0.0 # Hide this target
        else:
            mj_model.geom(self.target_geom_name).rgba[-1] = 1.0 # Show this target
            mj_model.body_pos[self.target_body_id, :] = state.info[f'phase_{phase}/target_pos']
            mj_model.geom_size[self.target_geom_id, :] = state.info[f'phase_{phase}/target_size']

class MyoUserUniversal(MyoUserBase): 
    
    def add_task_relevant_geoms(self, spec: mujoco.MjSpec):
        targets = self._config.task_config.targets
        self.target_objs = []
        rng = jax.random.PRNGKey(1)
        for i, target in enumerate(targets):
            rng, rng_init = jax.random.split(rng, 2)
            target_obj = PointingTargetClass(phase_number=i,
                target_pos_range=jp.array(target['position']),
                target_radius_range=jp.array(target['size']),
            )
            spec = target_obj.add_to_spec(spec, rng_init, target['rgb'])
            self.target_objs.append(target_obj)
        print(f"Added {len(self.target_objs)} targets to spec")
        return spec

    def preprocess_spec(self, spec: mujoco.MjSpec):
        spec = self.add_task_relevant_geoms(spec)
        return super().preprocess_spec(spec)

    def _setup(self):
        super()._setup()
        self.reach_settings = self._config.task_config.reach_settings
        self.obs_keys = self._config.task_config.obs_keys
        self.omni_keys = self._config.task_config.omni_keys
        
        if not self._config.vision.enabled:
            print(f"No vision, so adding {self.omni_keys} to obs_keys")
            self.obs_keys.extend(self.omni_keys)
        else:
            print(f"Vision, so not adding {self.omni_keys} to obs_keys")
        print(f"Obs keys: {self.obs_keys}")

    def _prepare_after_init(self, data):
        super()._prepare_after_init(data)
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.reach_settings.ref_site)].copy() + jp.array(self.reach_settings.target_origin_rel)  #jp.zeros(3,)
        for target_obj in self.target_objs:
            target_obj.target_coordinates_origin = self.target_coordinates_origin
            target_obj.target_body_id = self.mj_model.body(target_obj.target_body_name).id
            target_obj.target_geom_id = self.mj_model.geom(target_obj.target_geom_name).id

    def auto_reset(self):
        pass

    def reset(self, rng, render_token=None, add_to_info=None):
        _, rng = jax.random.split(rng, 2)

        data = self._reset_bm_model(rng)

        last_ctrl = jp.zeros(self._nu)

        info = {"rng": rng,
                "last_ctrl": last_ctrl,
                "steps_inside_target": jp.array(0.),
                'reach_dist': jp.array(0.),
                'target_success': jp.array(False),
                'target_fail': jp.array(False),
                'task_completed': jp.array(False),
                'phase': jp.array(0),
                'time_in_current_phase': jp.array(0.),
                }
        
        if add_to_info is not None:
            info.update(add_to_info)
        
        for target_obj in self.target_objs:
            update_info = target_obj.reset(info)
            info.update(update_info)
        
        info['target_pos'] = info['phase_0/target_pos']
        info['target_size'] = info['phase_0/target_size']        
        reward, done = jp.zeros(2)
        obs = None #TODO: implement this!
        metrics = {} #TODO: implement this!
        return State(data, obs, reward, done, metrics, info)


    def step(self, state: State, action: jp.ndarray) -> State:
        pass

    def update_task_visuals(self, mj_model, state):
        phase = state.info['phase']
        for target_obj in self.target_objs:
            target_obj.update_task_visuals(mj_model, state, phase)


        