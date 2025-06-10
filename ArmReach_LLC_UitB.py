import os
from datetime import datetime
from etils import epath
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

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
from brax.base import Base, Motion, Transform, System
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers.training import VmapWrapper, DomainRandomizationVmapWrapper, EpisodeWrapper, AutoResetWrapper
from myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v1 import AdaptiveTargetWrapper
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model

from matplotlib import pyplot as plt
import mediapy as media

def main(experiment_id='ArmReach', n_train_steps=20_000_000, n_eval_eps=10,
         restore_params_path=None, init_target_area_width_scale=0.):

  env_name = 'mobl_arms_index_llc_eepos_adaptive_mjx-v0'
  from myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v1 import LLCEEPosAdaptiveEnvMJXV0, LLCEEPosAdaptiveDirectCtrlEnvMJXV0
  envs.register_environment(env_name, LLCEEPosAdaptiveEnvMJXV0)
  
  model_path = 'simhive/uitb_sim/mobl_arms_index_llc_eepos_pointing.xml'
  path = (epath.Path(epath.resource_path('myosuite')) / (model_path)).as_posix()
  #TODO: load kwargs from config file/registration
  kwargs = {
            'frame_skip': 25,
            'target_pos_range': {'fingertip': jp.array([[0.225, -0.3, -0.3], [0.35, 0.1, 0.4]]),},
            'target_radius_range': {'fingertip': jp.array([0.01, 0.15]),},
            'ref_site': 'humphant',
            'adaptive_task': True,
            'init_target_area_width_scale': init_target_area_width_scale,
            # 'adaptive_increase_success_rate': 0.6,
            # 'adaptive_decrease_success_rate': 0.3,
            # 'adaptive_change_step_size': 0.05,
            # 'adaptive_change_min_trials': 50,
            'success_log_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform',
            # 'max_trials': 10
            'ctrl_dt': 0.002*25,
            'sim_dt': 0.002,
        }
  env = envs.get_environment(env_name, model_path=path, auto_reset=False, **kwargs)

  cwd = os.path.dirname(os.path.abspath(__file__))
  if restore_params_path is not None:
    restore_params = model.load_params(cwd + '/' + restore_params_path)
  else:
    restore_params = None


  def _render(rollouts, experiment_id='ArmReach', video_type='single', height=480, width=640, camera='for_testing'):

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

    os.makedirs(cwd + '/myosuite-mjx-evals/', exist_ok=True)
    eval_path = cwd + f'/myosuite-mjx-evals/{experiment_id}'

    if video_type == 'single':
      media.write_video(f'{eval_path}.mp4', videos, fps=1.0 / env.dt) 
    elif video_type == 'multiple':
      for i, video in enumerate(videos):
        media.write_video(f'{eval_path}_{i}.mp4', video, fps=1.0 / env.dt) 

    return None
  
  def _create_render(env, height=480, width=640, camera='for_testing'):
    front_view_pos = env.sys.mj_model.camera(camera).pos.copy()
    front_view_pos[1] = -2
    front_view_pos[2] = 0.65
    env.sys.mj_model.camera(camera).poscom0 = front_view_pos

    return functools.partial(env.render, height=height, width=width, camera=camera)
  
  train_fn = functools.partial(
      ppo.train, num_timesteps=n_train_steps, num_evals=0, reward_scaling=0.1,
      wrap_env_fn=wrap_curriculum_training, episode_length=800, #when wrap_curriculum_training is used, 'episode_length' only determines length of eval episodes
      normalize_observations=True, action_repeat=1,
      unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
      discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
      batch_size=512, seed=0,
      log_training_metrics=True,
      restore_params=restore_params,
      )
  ## rule of thumb: num_timesteps ~= (unroll_length * batch_size * num_minibatches) * [desired number of policy updates (internal variabele: "num_training_steps_per_epoch")]; Note: for fixed total env steps (num_timesteps), num_evals and num_resets_per_eval define how often policies are evaluated and the env is reset during training (split into Brax training "epochs")

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  x_data_train = []
  y_data_train = []
  y_data_train_length = []
  y_data_train_success = []
  y_data_train_curriculum_state = []

  max_y, min_y = 150, 0
  max_episode_length = 800
  def progress(num_steps, metrics):
    
    # print(metrics)

    if 'eval/episode_reward' in metrics:
      ## called during evaluation
      print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
          task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
          episode length: {metrics['eval/avg_episode_length']}")
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

      cwd = os.path.dirname(os.path.abspath(__file__))
      os.makedirs(cwd + '/myosuite-mjx-policies/', exist_ok=True)
      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{experiment_id}_progress.png')
      plt.savefig(fig_path)
    elif 'episode/sum_reward' in metrics:
      ## called during training
          # print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
          # task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
          # episode length: {metrics['eval/avg_episode_length']}")
      times.append(datetime.now())
      x_data_train.append(num_steps)
      y_data_train.append(metrics['episode/sum_reward'])
      y_data_train_length.append(metrics['episode/length'])
      y_data_train_success.append(metrics['episode/success_rate'])
      y_data_train_curriculum_state.append(metrics['episode/target_area_dynamic_width_scale'])

      cwd = os.path.dirname(os.path.abspath(__file__))
      os.makedirs(cwd + '/myosuite-mjx-policies/', exist_ok=True)

      ## Reward
      plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([min_y, max_y])

      plt.xlabel('# environment steps')
      plt.ylabel('reward per episode')
      plt.title(f'y={y_data_train[-1]:.3f}')

      plt.errorbar(
          x_data_train, y_data_train) # yerr=ydataerr)
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{experiment_id}_progress_train_reward.png')
      plt.savefig(fig_path)
      plt.close()

      ## Episode length
      plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([0, max_episode_length * 1.1])

      plt.xlabel('# environment steps')
      plt.ylabel('episode length')
      plt.title(f'length={y_data_train_length[-1]:.3f}')
      plt.errorbar(
          x_data_train, y_data_train_length) # yerr=ydataerr)
      plt.show()
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{experiment_id}_progress_train_length.png')
      plt.savefig(fig_path)
      plt.close()
      
      ## Success rate
      plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([0, 1])

      plt.xlabel('# environment steps')
      plt.ylabel('success rate')
      plt.title(f'success={y_data_train_success[-1]:.3f}')
      plt.errorbar(
          x_data_train, y_data_train_success) # yerr=ydataerr)
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{experiment_id}_progress_train_success.png')
      plt.savefig(fig_path)
      plt.close()
      
      ## Curriculum state
      plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([0, 1])

      plt.xlabel('# environment steps')
      plt.ylabel('curriculum state')
      plt.title(f'width={y_data_train_curriculum_state[-1]:.3f}')
      plt.errorbar(
          x_data_train, y_data_train_curriculum_state) # yerr=ydataerr)
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{experiment_id}_progress_train_curriculum.png')
      try:
        plt.savefig(fig_path)
        plt.close()
      except:
        pass

  ## TRAINING
  make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

  if n_train_steps > 0 and len(times) > 2:
    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

  cwd = os.path.dirname(os.path.abspath(__file__))
  os.makedirs(cwd + '/myosuite-mjx-policies/', exist_ok=True)

  param_path = os.path.join(cwd, f'myosuite-mjx-policies/{experiment_id}_params')
  if n_train_steps > 0:
    model.save_params(param_path, params)

  # TODO: store metrics as well
  print(metrics)

  ## EVALUATION
  ##TODO: load internal env state ('target_area_dynamic_width_scale')
  backend = 'positional' # @param ['generalized', 'positional', 'spring']
  eval_env = envs.create(env_name=env_name, model_path=path, eval_mode=True, backend=backend, episode_length=env._episode_length, **kwargs)

  params = model.load_params(param_path)
  inference_fn = make_inference_fn(params)

  times = [datetime.now()]

  render_fn = _create_render(env)
  rollouts = evaluate(eval_env, inference_fn, n_eps=n_eval_eps, times=times, render_fn=render_fn)

  _render(rollouts, experiment_id=experiment_id)

def wrap_curriculum_training(
    env: Env,
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
    if randomization_fn is None:
      env = VmapWrapper(env)
    else:
      env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AdaptiveTargetWrapper(env)

    return env

def evaluate(env, inference_fn, n_eps=10, rng=None, times=[], render_fn=None, video_type='single'):
  if rng is None:
    rng = jax.random.PRNGKey(seed=0)
  
  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(inference_fn)

  state = jit_env_reset(rng=rng)
  act = jp.zeros(env.sys.nu)
  state = jit_env_step(state, act)

  times.append(datetime.now())
  print(f'time to jit: {times[1] - times[0]}')

  rollouts = []
  videos = []
  for episode in range(n_eps):
    rng = jax.random.PRNGKey(seed=episode)
    state = jit_env_reset(rng=rng)
    rollout = {}
    # rollout['wrist_target'] = state.info['wrist_target']
    states = []
    states_pointer = 0
    frame_collection = []
    # while not (state.done or state.info['truncation']):
    done = state.done
    while not (done or state.info['truncation']):
      states.append(state.pipeline_state)
      act_rng, rng = jax.random.split(rng)
      act, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_env_step(state, act)
      
      obs_dict = env.get_obs_dict(state.pipeline_state, state.info)
      done = obs_dict['task_completed']
      if obs_dict['target_success'] or obs_dict['target_fail']:
        # print('RENDER', rollout_pointer, len(rollout), obs_dict['target_pos'])
        if render_fn is not None:
          frame_collection.extend(render_fn(states[states_pointer:], state.info['target_pos'])) #with render_mode="rgb_array_list", env.render() returns a list of all frames since last call of reset()
          states_pointer = len(states)

    times.append(datetime.now())

    rollout['states'] = states
    rollouts.append(rollout)

    if render_fn is not None:
      frame_collection.extend(render_fn(states[states_pointer:], state.info['target_pos'])) #with render_mode="rgb_array_list", env.render() returns a list of all frames since last call of reset()
      states_pointer = len(states)
      videos += frame_collection
  
  cwd = os.path.dirname(os.path.abspath(__file__))
  os.makedirs(cwd + '/myosuite-mjx-evals/', exist_ok=True)
  eval_path = cwd + f'/myosuite-mjx-evals/{experiment_id}'

  if video_type == 'single':
    media.write_video(f'{eval_path}.mp4', videos, fps=1.0 / env.dt) 
  elif video_type == 'multiple':
    for i, video in enumerate(videos):
      media.write_video(f'{eval_path}_{i}.mp4', video, fps=1.0 / env.dt) 

  return rollouts


if __name__ == '__main__':
  # jax.config.update('jax_default_matmul_precision', 'highest')

  experiment_id = 'schema_change_final'
  n_train_steps = 100_000_000
  n_eval_eps = 1

  restore_params_path = None  #"myosuite-mjx-policies/mobl_llc_eepos_v0.1.1b_params"
  init_target_area_width_scale = 0.  #1.0  #TODO: load from file

  main(experiment_id=experiment_id, n_train_steps=n_train_steps, n_eval_eps=n_eval_eps, 
       restore_params_path=restore_params_path, init_target_area_width_scale=init_target_area_width_scale)