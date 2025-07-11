# Modified from code in Mujoco Playground to suit this project
#
#
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
"""Wrappers for MuJoCo Playground environments."""

import functools
from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers.training import VmapWrapper, DomainRandomizationVmapWrapper, EpisodeWrapper, AutoResetWrapper
from brax.base import System
from brax import base

from mujoco_playground._src import mjx_env

class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: PipelineEnv,
      randomization_fn: Callable[[base.System], Tuple[base.System, base.System]],
  ):
    super().__init__(env)
    self._sys_v, self._in_axes = randomization_fn(self.env.sys)

  def _env_fn(self, sys: base.System) -> PipelineEnv:
    env = self.env
    # This might cause a problem
    env.unwrapped.sys = sys
    return env

  def reset(self, rng: jax.Array) -> State:
    def reset(sys, rng):
      env = self._env_fn(sys=sys)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(sys, s, a):
      env = self._env_fn(sys=sys)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._sys_v, state, action
    )
    return res


def _identity_vision_randomization_fn(
    sys: base.System, num_worlds: int
) -> Tuple[base.System, base.System]:
  """Tile the necessary fields for the Madrona memory buffer copy."""
  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_directional': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  sys = sys.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(sys.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(sys.geom_matid, 0), num_worlds, axis=0
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(sys.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(sys.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(sys.light_dir, 0), num_worlds, axis=0
      ),
      'light_directional': jp.repeat(
          jp.expand_dims(sys.light_directional, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(sys.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(sys.light_cutoff, 0), num_worlds, axis=0
      ),
  })
  return sys, in_axes


def _supplement_vision_randomization_fn(
    sys: base.System,
    randomization_fn: Callable[[base.System], Tuple[base.System, base.System]],
    num_worlds: int,
) -> Tuple[base.System, base.System]:
  """Tile the necessary missing fields for the Madrona memory buffer copy."""
  sys, in_axes = randomization_fn(sys)

  required_fields = [
      'geom_rgba',
      'geom_matid',
      'geom_size',
      'light_pos',
      'light_dir',
      'light_directional',
      'light_castshadow',
      'light_cutoff',
  ]

  for field in required_fields:
    if getattr(in_axes, field) is None:
      in_axes = in_axes.tree_replace({field: 0})
      val = getattr(sys, field)
      sys = sys.tree_replace({
          field: jp.repeat(jp.expand_dims(val, 0), num_worlds, axis=0),
      })
  return sys, in_axes


class MadronaWrapper:
  """Wraps a MuJoCo Playground to be used in Brax with Madrona."""

  def __init__(
      self,
      env: PipelineEnv,
      num_worlds: int,
      randomization_fn: Optional[
          Callable[[base.System], Tuple[base.System, base.System]]
      ] = None,
  ):
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_vision_randomization_fn, num_worlds=num_worlds
      )
    else:
      randomization_fn = functools.partial(
          _supplement_vision_randomization_fn,
          randomization_fn=randomization_fn,
          num_worlds=num_worlds,
      )
    self._env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    self.num_worlds = num_worlds

    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_directional',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._sys_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    return self._env.reset(rng)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    return self._env.step(state, action)

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped instance."""
    return getattr(self._env.unwrapped, name)


def wrap_curriculum_training(
    env: Env,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System]]
    ] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    if vision:
      env = MadronaWrapper(env, num_worlds=num_vision_envs, randomization_fn=randomization_fn)
    elif randomization_fn is None:
      env = VmapWrapper(env)
    else:
      env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    return env