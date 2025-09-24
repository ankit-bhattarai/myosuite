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
from myosuite.envs.myo.myouser.myouser_pointing_v0 import MyoUserPointing, PointingTaskConfig
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class TrackingTaskConfig(PointingTaskConfig):
    planar_x : bool = True
    num_components: int = 5
    min_amplitude: float = 1
    max_amplitude: float = 5
    min_frequency: float = 0.0
    max_frequency: float = 0.5
    max_episode_steps: int = 40
    omni_keys: List[str] = field(default_factory=lambda: ["target_pos", "prev_target_pos", "prev_prev_target_pos", "target_radius"]) 

@dataclass
class TrackingEnvConfig(BaseEnvConfig):
    env_name: str = "MyoUserTracking"
    model_path: str = "myosuite/envs/myo/assets/arm/mobl_arms_index_reaching_myouser.xml"
    task_config: TrackingTaskConfig = field(default_factory=lambda: TrackingTaskConfig())

class MyoUserTracking(MyoUserPointing):    
    def _setup(self):
        """Task specific setup"""
        super()._setup()
        self.task_config = self._config.task_config

    @staticmethod
    def generate_sine_wave(rng, limits, num_components, min_amplitude, max_amplitude, min_frequency, max_frequency, max_episode_steps, dt):
        t = jp.arange(max_episode_steps) * dt
        rng_amp, rng_freq, rng_phase = jax.random.split(rng, 3)

        amplitudes = jax.random.uniform(rng_amp, shape=(num_components,), 
                                minval=min_amplitude, maxval=max_amplitude)
        frequencies = jax.random.uniform(rng_freq, shape=(num_components,), 
                                        minval=min_frequency, maxval=max_frequency)
        phases = jax.random.uniform(rng_phase, shape=(num_components,), 
                                minval=0, maxval=2*jp.pi)

        sine_components = amplitudes[:, None] * jp.sin(
        frequencies[:, None] * 2 * jp.pi * t[None, :] + phases[:, None]
    )

        # Sum all components to get final sine wave
        sine = jp.sum(sine_components, axis=0)
        sum_amplitude = jp.sum(amplitudes)

        sine = (sine + sum_amplitude) / (2*sum_amplitude)
        sine = limits[0] + (limits[1] - limits[0])*sine
        return sine

    def generate_trajectory(self, rng):
        num_components = self.task_config.num_components
        min_amplitude = self.task_config.min_amplitude
        max_amplitude = self.task_config.max_amplitude
        min_frequency = self.task_config.min_frequency
        max_frequency = self.task_config.max_frequency
        max_episode_steps = self.max_steps_without_hit
        target_pos_range = self.reach_settings.target_pos_range.items()
        assert len(target_pos_range) == 1, "Only one target pos range is supported"
        site, span = target_pos_range[0]
        x_low, x_high = span[0][0], span[1][0]
        y_low, y_high = span[0][1], span[1][1]
        z_low, z_high = span[0][2], span[1][2]
        dt = self.dt
        x_rng, y_rng, z_rng = jax.random.split(rng, 3)
        size_trajectory = int(max_episode_steps)+3
        if self.task_config.planar_x:
            assert x_low == x_high, "x_low and x_high must be the same for planar_x"
            xs = x_low * jp.ones(size_trajectory)
        else:    
            xs = self.generate_sine_wave(x_rng, [x_low, x_high], num_components, min_amplitude, max_amplitude, min_frequency, max_frequency, max_episode_steps, dt)
        
        xs = self.target_coordinates_origin[0] + xs
        ys = self.generate_sine_wave(y_rng, [y_low, y_high], num_components, min_amplitude, max_amplitude, min_frequency, max_frequency, size_trajectory, dt)
        ys = self.target_coordinates_origin[1] + ys
        zs = self.generate_sine_wave(z_rng, [z_low, z_high], num_components, min_amplitude, max_amplitude, min_frequency, max_frequency, size_trajectory, dt)
        zs = self.target_coordinates_origin[2] + zs
        return jp.stack([xs, ys, zs], axis=1)
        
    def reset(self, rng, target_pos=None, target_radius=None, render_token=None):
        trajectory_rng, rng = jax.random.split(rng, 2)
        trajectory = self.generate_trajectory(trajectory_rng)
        add_to_info = {
            'prev_prev_target_pos': trajectory[0],
            'prev_target_pos': trajectory[1],
            'trajectory': trajectory,
            'step_index': 0,
        }
        state = super().reset(rng, target_pos=trajectory[2], add_to_info=add_to_info)
        return state
    
    def step(self, state: State, action: jp.ndarray) -> State:
        trajectory = state.info['trajectory']
        step_index = state.info['step_index']
        state.info['step_index'] = step_index + 1
        state.info['target_pos'] = trajectory[step_index + 3]
        state.info['prev_target_pos'] = trajectory[step_index + 2]
        state.info['prev_prev_target_pos'] = trajectory[step_index + 1]
        return super().step(state, action)

    def get_obs_dict(self, data, info):
        obs_dict = super().get_obs_dict(data, info)
        obs_dict['prev_target_pos'] = info['prev_target_pos']
        obs_dict['prev_prev_target_pos'] = info['prev_prev_target_pos']
        return obs_dict
    
