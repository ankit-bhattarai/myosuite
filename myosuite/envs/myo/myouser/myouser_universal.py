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

class MyoUserUniversal(MyoUserBase): 
    
    def add_task_relevant_geoms(self, spec: mujoco.MjSpec):
        targets = self._config.task_config.targets
        worldbody = spec.worldbody
        key = jax.random.PRNGKey(1)
        target_body_names = []
        target_geom_names = []
        for i, target in enumerate(targets):
            assert target['shape'] == "sphere", "Only sphere shapes are supported for now"
            target_body_name = f"body_target_{i}"
            key, pos_key, size_key = jax.random.split(key, 3)
            target_pos = jax.random.uniform(pos_key, (3,), minval=jp.array(target['position'][0]), maxval=jp.array(target['position'][1]))
            target_size = jax.random.uniform(size_key, (3,), minval=jp.array(target['size'][0]), maxval=jp.array(target['size'][1]))
            target_body = worldbody.add_body(
                name=target_body_name,
                pos=target_pos,
            )
            target_body_names.append(target_body_name)
            target_geom_name = f"geom_target_{i}"
            rgba = jp.zeros(4)
            rgba = rgba.at[:3].set(target['rgb'])
            target_geom = target_body.add_geom(
                name=target_geom_name,
                pos=jp.zeros(3),
                size=target_size,
                rgba=rgba,
            )
            target_geom_names.append(target_geom_name)

        self.target_body_names = target_body_names
        self.target_geom_names = target_geom_names
        print("Adding task relevant geoms")
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

        self.target_body_ids = jp.array([self.mj_model.body(name).id for name in self.target_body_names])
        self.target_geom_ids = jp.array([self.mj_model.geom(name).id for name in self.target_geom_names])

    def _prepare_after_init(self, data):
        super()._prepare_after_init(data)
        # Define target origin, relative to which target positions will be generated
        self.target_coordinates_origin = data.site_xpos[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.reach_settings.ref_site)].copy() + jp.array(self.reach_settings.target_origin_rel)  #jp.zeros(3,)

    def get_generate_target(self, rng, phase=0):
        target = self._config.task_config.targets[phase]
        rng, pos_rng, size_rng = jax.random.split(rng, 3)
        target_pos = jax.random.uniform(pos_rng, (3,), minval=jp.array(target['position'][0]), maxval=jp.array(target['position'][1]))
        target_size = jax.random.uniform(size_rng, (3,), minval=jp.array(target['size'][0]), maxval=jp.array(target['size'][1]))
        target_pos = self.target_coordinates_origin + target_pos
        return target_pos, target_size

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
        target_pos, target_size = self.get_generate_target(rng, phase=0)
        info['target_pos'] = target_pos
        info['target_size'] = target_size
        reward, done = jp.zeros(2)
        obs = None #TODO: implement this!
        metrics = {} #TODO: implement this!
        return State(data, obs, reward, done, metrics, info)


    def step(self, state: State, action: jp.ndarray) -> State:
        pass

    def update_task_visuals(self, mj_model, state):
        phase = state.info['phase']

        # Show target from this phase and hide all others
        target_body_id = self.target_body_ids[phase]
        target_geom_id = self.target_geom_ids[phase]

        for i in range(len(self._config.task_config.targets)):
            mj_model.geom(self.target_geom_ids[i]).rgba[-1] = 0.0
        mj_model.geom(target_geom_id).rgba[-1] = 1.0

        mj_model.body_pos[target_body_id, :] = state.info['target_pos']
        mj_model.geom_size[target_geom_id, :] = state.info['target_size']

        