from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union

import jax
from jax import numpy as jp
import numpy as np
from jax import nn

import mujoco
from mujoco import mjx
from myosuite.utils import gym

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model

from matplotlib import pyplot as plt
import mediapy as media

def main():

  env_name = 'myoArm_llc_eepos_mjx-v0'
  from myosuite.envs.myo.myouser.llc_eepos_mjx_v0 import LLCEEPosEnvMJXV0, LLCEEPosDirectCtrlEnvMJXV0
  envs.register_environment(env_name, LLCEEPosDirectCtrlEnvMJXV0)
  
  model_path = 'simhive/uitb_sim/mobl_arms_index_llc_eepos_pointing.xml'
  path = (epath.Path(epath.resource_path('myosuite')) / (model_path)).as_posix()
  #TODO: load kwargs from config file/registration
  kwargs = {
            'frame_skip': 25,
            'target_pos_range': {'fingertip': jp.array([[0.225, -0.3, -0.3], [0.35, 0.1, 0.4]]),},
            'target_radius_range': {'fingertip': jp.array([0.01, 0.15]),},
            'ref_site': 'humphant',
            'adaptive_task': True,
            # 'init_target_area_width_scale': 0,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'success_log_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform',
            # 'max_trials': 1
        }
  env = envs.get_environment(env_name, model_path=path, auto_reset=False, **kwargs)

  def _render(rollouts, video_type='single', height=480, width=640, camera='for_testing'):

    front_view_pos = env.sys.mj_model.camera(camera).pos.copy()
    front_view_pos[1] = -2
    front_view_pos[2] = 0.65
    env.sys.mj_model.camera(camera).poscom0 = front_view_pos

    videos = []
    for rollout in rollouts:
      
      # change the target position of the environment for rendering
      ## TODO: generate new target for each rollout
      # env.sys.mj_model.site_pos[env._target_sids] = rollout['wrist_target']

      if video_type == 'single':
        videos += env.render(rollout['states'], height=height, width=width, camera=camera)
      elif video_type == 'multiple':
        videos.append(env.render(rollout['states'], height=height, width=width, camera=camera))

    if video_type == 'single':
      media.write_video(cwd + '/ArmReach.mp4', videos, fps=1.0 / env.dt) 
    elif video_type == 'multiple':
      for i, video in enumerate(videos):
        media.write_video(cwd + '/ArmReach' + str(i) + '.mp4', video, fps=1.0 / env.dt) 

    return None
  
  train_fn = functools.partial(
      ppo.train, num_timesteps=20_000_000, num_evals=20, reward_scaling=0.1,
      episode_length=800, normalize_observations=True, action_repeat=1,
      unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
      discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
      batch_size=512, seed=0)
  ## rule of thumb: num_timesteps ~= (unroll_length * batch_size * num_minibatches) * [desired number of policy updates (internal variabele: "num_training_steps_per_epoch")]; Note: for fixed total env steps (num_timesteps), num_evals and num_resets_per_eval define how often policies are evaluated and the env is reset during training (split into Brax training "epochs")

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  max_y, min_y = 13000, 0
  def progress(num_steps, metrics):

    print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}")

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    plt.show()

  make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')

  import os
  cwd = os.path.dirname(os.path.abspath(__file__))

  model.save_params(cwd + '/ArmReachParams', params)
  params = model.load_params(cwd + '/ArmReachParams')
  inference_fn = make_inference_fn(params)

  backend = 'positional' # @param ['generalized', 'positional', 'spring']
  env = envs.create(env_name=env_name, model_path=path, backend=backend, episode_length=env._episode_length, **kwargs)


  times = [datetime.now()]

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(inference_fn)

  rng = jax.random.PRNGKey(seed=0)
  state = jit_env_reset(rng=rng)
  act = jp.zeros(env.sys.nu)
  state = jit_env_step(state, act)

  times.append(datetime.now())
  print(f'time to jit: {times[1] - times[0]}')

  rollouts = []
  for episode in range(10):
    rng = jax.random.PRNGKey(seed=episode)
    state = jit_env_reset(rng=rng)
    rollout = {}
    # rollout['wrist_target'] = state.info['wrist_target']
    states = []
    while not (state.done or state.info['truncation']):
      states.append(state.pipeline_state)
      act_rng, rng = jax.random.split(rng)
      act, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_env_step(state, act)

    times = [datetime.now()]

    rollout['states'] = states
    rollouts.append(rollout)

  _render(rollouts)

if __name__ == '__main__':
  # jax.config.update('jax_default_matmul_precision', 'highest')

  main()