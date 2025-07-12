import os
import argparse

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"]="egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

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
from flax import linen
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform, System
from brax.base import State as PipelineState
from myosuite.train.utils.wrapper import wrap_curriculum_training
from myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v1 import AdaptiveTargetWrapper
from brax.mjx.base import State as MjxState
# from brax.training.agents.ppo import train as ppo
from myosuite.train.myouser.custom_ppo import train as ppo
from myosuite.train.myouser.custom_ppo import networks_vision_unified as networks
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model
from myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v1 import LLCEEPosAdaptiveEnvMJXV0

from matplotlib import pyplot as plt
import mediapy as media
import wandb

class ProgressLogger:
  def __init__(self):
    self.times = [datetime.now()]

  def progress(self, num_steps, metrics):
    self.times.append(datetime.now())
    if len(self.times) == 2:
        print(f'time to jit: {self.times[1] - self.times[0]}')
    wandb.log({'num_steps': num_steps, **metrics})

def main(experiment_id, project_id='mjx-training', n_train_steps=100_000_000, n_eval_eps=1,
         restore_params_path=None, init_target_area_width_scale=0.,
         num_envs=3072,
         vision=True,
         vision_mode='rgbd',
         activation_function='swish',
         policy_hidden_layer_sizes=(256, 256),
         value_hidden_layer_sizes=(256, 256),
         episode_length=800,
         unroll_length=10,
         num_minibatches=24,
         num_updates_per_batch=8,
         discounting=0.97,
         learning_rate=3e-4,
         entropy_cost=1e-3,
         batch_size=512,
         target_pos_range_min=[0.225, -0.1, -0.3],
         target_pos_range_max=[0.35, 0.1, 0.3],
         adaptive_increase_success_rate=0.8,
         adaptive_decrease_success_rate=0.5,
         adaptive_change_step_size=0.05,
         adaptive_change_min_trials=100,
         vision_output_size=20,
         weights_reach=1.0,
         weights_bonus=8.0,
         reach_metric_coefficient=10.0,
         get_env_only=False,
         cheat_vision_aux_output=False,
         ):

  env_name = 'mobl_arms_index_llc_eepos_adaptive_mjx-v0'
  envs.register_environment(env_name, LLCEEPosAdaptiveEnvMJXV0)
  
  model_path = 'simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml'
  path = (epath.Path(epath.resource_path('myosuite')) / (model_path)).as_posix()
  #TODO: load kwargs from config file/registration

  argument_kwargs = {
    'experiment_id': experiment_id,
    'n_train_steps': n_train_steps,
    'n_eval_eps': n_eval_eps,
    'restore_params_path': restore_params_path,
    'init_target_area_width_scale': init_target_area_width_scale,
    'num_envs': num_envs,
    'vision': vision,
    'vision_mode': vision_mode,
    'vision_output_size': vision_output_size,
    'activation_function': activation_function,
    'policy_hidden_layer_sizes': policy_hidden_layer_sizes,
    'value_hidden_layer_sizes': value_hidden_layer_sizes,
    'episode_length': episode_length,
    'unroll_length': unroll_length,
    'num_minibatches': num_minibatches,
    'num_updates_per_batch': num_updates_per_batch,
    'discounting': discounting,
    'learning_rate': learning_rate,
    'entropy_cost': entropy_cost,
    'batch_size': batch_size,
    'cheat_vision_aux_output': cheat_vision_aux_output,
  }
  kwargs = {
            'frame_skip': 25,
            'target_pos_range': {'fingertip': jp.array([target_pos_range_min, target_pos_range_max]),},
            'target_radius_range': {'fingertip': jp.array([0.05, 0.05]),},
            'ref_site': 'humphant',
            'adaptive_task': True,
            'init_target_area_width_scale': init_target_area_width_scale,
            'adaptive_increase_success_rate': adaptive_increase_success_rate,
            'adaptive_decrease_success_rate': adaptive_decrease_success_rate,
            'adaptive_change_step_size': adaptive_change_step_size,
            'adaptive_change_min_trials': adaptive_change_min_trials,
            'success_log_buffer_length': 500,
            # 'normalize_act': True,
            'reset_type': 'range_uniform',
            # 'max_trials': 10
            'num_envs': num_envs,
            'weights/reach': weights_reach,
            'weights/bonus': weights_bonus,
            'reach_metric_coefficient': reach_metric_coefficient,
        }
  if vision:
    kwargs['vision'] = {
                'gpu_id': 0,
                'render_width': 120,
                'render_height': 120,
                'enabled_cameras': [0],
                'vision_mode': vision_mode,
            }
  all_config = {**kwargs, **argument_kwargs}
  wandb.init(project=project_id, name=experiment_id, config=all_config)
  env = envs.get_environment(env_name, model_path=path, auto_reset=False, **kwargs)

  cwd = os.path.dirname(os.path.abspath(__file__))
  if restore_params_path is not None:
    restore_params = model.load_params(cwd + '/' + restore_params_path)
    print(f'Restored params from {restore_params_path}')
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
  
  def get_observation_size():
    if 'vision' not in kwargs:
      return {'proprioception': 48}
    if vision_mode == 'rgb':
      return {
          "pixels/view_0": (120, 120, 3),  # RGB image
          "proprioception": (44,)          # Vector state
          }
    elif vision_mode == 'rgbd':
      return {
          "pixels/view_0": (120, 120, 4),  # RGBD image
          "proprioception": (44,)          # Vector state
      }
    elif vision_mode == 'rgb+depth':
      return {
          "pixels/view_0": (120, 120, 3),  # RGB image
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (44,)          # Vector state
          }
    elif vision_mode == 'rgbd_only':
      return {
          "pixels/view_0": (120, 120, 4),  # RGBD image
      }
    elif vision_mode == 'depth_only':
      return {
          "pixels/depth": (120, 120, 1),  # Depth image
      }
    elif vision_mode == 'depth':
      return {
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (44,)          # Vector state
      }
    elif vision_mode == 'depth_w_aux_task':
      return {
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (44,),          # Vector state
          "vision_aux_targets": (4,) # 3D target position + 1D target radius
      }
    else:
      raise NotImplementedError(f'No observation size known for "{vision_mode}"')

  def custom_network_factory(obs_shape, action_size, preprocess_observations_fn):
      if activation_function == 'swish':
        activation = linen.swish
      elif activation_function == 'relu':
        activation = linen.relu
      else:
        raise NotImplementedError(f'Not implemented anything for activation function {activation_function}')
      # if not vision:
      #   return networks.make_ppo_networks_no_vision(
      #     proprioception_size=get_observation_size()['proprioception'],
      #     action_size=action_size,
      #     preprocess_observations_fn=preprocess_observations_fn,
      #   )
      return networks.make_ppo_networks_with_vision(
        proprioception_size=44,
        action_size=action_size,
        encoder_out_size=4,
        preprocess_observations_fn=preprocess_observations_fn,
        cheat_vision_aux_output=True,
      )
      # return networks.make_ppo_networks_unified_extractor(
      #   observation_size=get_observation_size(),
      #   action_size=action_size,
      #   preprocess_observations_fn=preprocess_observations_fn,
      #   policy_hidden_layer_sizes=policy_hidden_layer_sizes,
      #   value_hidden_layer_sizes=value_hidden_layer_sizes,
      #   vision_output_size=vision_output_size,
      #   activation=activation,
      #   normalise_pixels=True,
      # )
      # if vision:
      #   return networks_vision.make_ppo_networks_vision(
      #       observation_size=get_observation_size(),
      #       action_size=action_size,
      #       preprocess_observations_fn=preprocess_observations_fn,
      #       policy_hidden_layer_sizes=policy_hidden_layer_sizes,  
      #       value_hidden_layer_sizes=value_hidden_layer_sizes,
      #       activation=activation,
      #       normalise_channels=True            # Normalize image channels
      #   )
      # else:
      #   return networks.make_ppo_networks(observation_size=get_observation_size(),
      #                                     action_size=action_size,
      #                                     preprocess_observations_fn=preprocess_observations_fn,
      #                                     policy_hidden_layer_sizes=policy_hidden_layer_sizes,
      #                                     value_hidden_layer_sizes=value_hidden_layer_sizes,
      #                                     activation=activation)

  train_fn = functools.partial(
      ppo.train, num_timesteps=n_train_steps, num_evals=0, reward_scaling=0.1,
      madrona_backend=vision,
      wrap_env=False, episode_length=episode_length, #when wrap_curriculum_training is used, 'episode_length' only determines length of eval episodes
      normalize_observations=True, action_repeat=1,
      unroll_length=unroll_length, num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
      discounting=discounting, learning_rate=learning_rate, entropy_cost=entropy_cost, num_envs=kwargs['num_envs'],
      num_eval_envs=kwargs['num_envs'],
      batch_size=batch_size, seed=0,
      log_training_metrics=True,
      restore_params=restore_params,
      network_factory=custom_network_factory,
      )
  ## rule of thumb: num_timesteps ~= (unroll_length * batch_size * num_minibatches) * [desired number of policy updates (internal variabele: "num_training_steps_per_epoch")]; Note: for fixed total env steps (num_timesteps), num_evals and num_resets_per_eval define how often policies are evaluated and the env is reset during training (split into Brax training "epochs")



  ## TRAINING
  wrapped_env = wrap_curriculum_training(env, vision=True, num_vision_envs=kwargs['num_envs'], episode_length=episode_length)
  # Adding custom task specific adaptive target wrapper
  wrapped_env = AdaptiveTargetWrapper(wrapped_env)
  if get_env_only:
    return wrapped_env
  
  progress_logger = ProgressLogger()
  make_inference_fn, params, metrics = train_fn(environment=wrapped_env, progress_fn=progress_logger.progress)
  times = progress_logger.times
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
  parser = argparse.ArgumentParser(description='ArmReach LLC UitB Training Script')
  parser.add_argument('--experiment_id', type=str, required=True)
  parser.add_argument('--project_id', type=str, default='mjx-training')
  parser.add_argument('--n_train_steps', type=int, default=100_000_000)
  parser.add_argument('--n_eval_eps', type=int, default=1)
  parser.add_argument('--restore_params_path', type=str, default=None)
  parser.add_argument('--init_target_area_width_scale', type=float, default=0.)
  parser.add_argument('--num_envs', type=int, default=3072)
  parser.add_argument('--no-vision', dest='vision', action='store_false', help='Type this if you want to disable vision input, default is true')
  parser.set_defaults(vision=True)
  parser.add_argument('--vision_mode', type=str, default='rgbd', help='Change to rgb or rgb+depth if wanting to change vision mode')
  parser.add_argument('--activation_function', type=str, default='swish', choices=('relu', 'swish',),
                      help='Choose between one of these two activation functions')
  parser.add_argument('--policy_hidden_layer_sizes', type=int, nargs='+', default=[256, 256])
  parser.add_argument('--value_hidden_layer_sizes', type=int, nargs='+', default=[256, 256])
  parser.add_argument('--episode_length', type=int, default=800)
  parser.add_argument('--unroll_length', type=int, default=10)
  parser.add_argument('--num_minibatches', type=int, default=24)
  parser.add_argument('--num_updates_per_batch', type=int, default=8)
  parser.add_argument('--discounting', type=float, default=0.97)
  parser.add_argument('--learning_rate', type=float, default=3e-4)
  parser.add_argument('--entropy_cost', type=float, default=1e-3)
  parser.add_argument('--batch_size', type=int, default=512)
  parser.add_argument('--target_pos_range_min', type=float, nargs='+', default=[0.225, -0.1, -0.3])
  parser.add_argument('--target_pos_range_max', type=float, nargs='+', default=[0.35, 0.1, 0.3])
  parser.add_argument('--adaptive_increase_success_rate', type=float, default=0.8)
  parser.add_argument('--adaptive_decrease_success_rate', type=float, default=0.5)
  parser.add_argument('--adaptive_change_step_size', type=float, default=0.05)
  parser.add_argument('--adaptive_change_min_trials', type=int, default=100)
  parser.add_argument('--vision_output_size', type=int, default=20)
  parser.add_argument('--weights_reach', type=float, default=1.0)
  parser.add_argument('--weights_bonus', type=float, default=8.0)
  parser.add_argument('--reach_metric_coefficient', type=float, default=10.0)
  parser.add_argument('--cheat_vision_aux_output', type=bool, default=False)
  args = parser.parse_args()

  main(
    experiment_id=args.experiment_id,
    project_id=args.project_id,
    n_train_steps=args.n_train_steps,
    n_eval_eps=args.n_eval_eps,
    restore_params_path=args.restore_params_path,
    init_target_area_width_scale=args.init_target_area_width_scale,
    num_envs=args.num_envs,
    vision=args.vision,
    vision_mode=args.vision_mode,
    activation_function=args.activation_function,
    policy_hidden_layer_sizes=tuple(args.policy_hidden_layer_sizes),
    value_hidden_layer_sizes=tuple(args.value_hidden_layer_sizes),
    episode_length=args.episode_length,
    unroll_length=args.unroll_length,
    num_minibatches=args.num_minibatches,
    num_updates_per_batch=args.num_updates_per_batch,
    discounting=args.discounting,
    learning_rate=args.learning_rate,
    entropy_cost=args.entropy_cost,
    batch_size=args.batch_size,
    target_pos_range_min=args.target_pos_range_min,
    target_pos_range_max=args.target_pos_range_max,
    adaptive_increase_success_rate=args.adaptive_increase_success_rate,
    adaptive_decrease_success_rate=args.adaptive_decrease_success_rate,
    adaptive_change_step_size=args.adaptive_change_step_size,
    adaptive_change_min_trials=args.adaptive_change_min_trials,
    vision_output_size=args.vision_output_size,
    weights_reach=args.weights_reach,
    weights_bonus=args.weights_bonus,
    reach_metric_coefficient=args.reach_metric_coefficient,
    cheat_vision_aux_output=args.cheat_vision_aux_output,
  )