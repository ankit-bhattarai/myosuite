# %%
# @title Configuration for both local and for Colab instances.

# On your second reading, load the compiled rendering backend to save time!
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MADRONA_MWGPU_KERNEL_CACHE"]="/home/ab2731/rds/hpc-work/madrona/madrona_mjx/kernel_cache"
os.environ["MADRONA_BVH_KERNEL_CACHE"]="/home/ab2731/rds/hpc-work/madrona/madrona_mjx/bvh_cache"
# os.environ["JAX_COMPILATION_CACHE_DIR"] = "/scratch/ankit/jax_cache"
# os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = "-1"
# os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = "0"
# os.environ['JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES'] = 'xla_gpu_kernel_cache_file'

# Check if MuJoCo installation was successful
import distutils.util
import os
import subprocess
if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError(
        'Cannot communicate with GPU. '
        'Make sure you are using a GPU Colab runtime. '
        'Go to the Runtime menu and select Choose runtime type.'
    )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
        f.write("""{
        "file_format_version" : "1.0.0",
        "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
        }
    }
    """)

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'
try:
    print('Checking that the installation succeeded:')
    import mujoco

    mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
    raise e from RuntimeError(
        'Something went wrong during installation. Check the shell output above '
        'for more information.\n'
        'If using a hosted Colab runtime, make sure you enable GPU acceleration '
        'by going to the Runtime menu and selecting "Choose runtime type".'
    )

print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# %%
import os
from datetime import datetime
from etils import epath
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jp
import numpy as np
from jax import nn
import wandb

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
from brax.training.acme import specs
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model
import argparse

from matplotlib import pyplot as plt
import mediapy as media
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from mujoco_playground._src import wrapper
from brax.envs.wrappers.training import VmapWrapper, DomainRandomizationVmapWrapper, EpisodeWrapper, AutoResetWrapper
from myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v1 import AdaptiveTargetWrapper
from brax.training.agents.ppo import networks_vision 
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from brax.base import Base, Motion, Transform, System
from brax.v1 import envs as envs_v1
from brax.training.acme import running_statistics


# %%
wrap_for_brax_training = wrapper.wrap_for_brax_training

# %%
class Progress:
  def __init__(self, train_fn, experiment_id):
    self.x_data = []
    self.y_data = []
    self.ydataerr = []
    self.times = [datetime.now()]

    self.x_data_train = []
    self.y_data_train = []
    self.y_data_train_length = []
    self.y_data_train_success = []
    self.y_data_train_curriculum_state = []

    self.max_y, self.min_y = 150, 0
    self.max_episode_length = 800
    self.train_fn = train_fn
    self.experiment_id = experiment_id

  def progress(self, num_steps, metrics):
    # print(metrics)
    if len(self.times) == 2:
       print(f'time to jit: {self.times[1] - self.times[0]}')
    wandb.log({'num_steps': num_steps, **metrics})
    if 'eval/episode_reward' in metrics:
      ## called during evaluation
      print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
          task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
          episode length: {metrics['eval/avg_episode_length']}")
      self.times.append(datetime.now())
      self.x_data.append(num_steps)
      self.y_data.append(metrics['eval/episode_reward'])
      self.ydataerr.append(metrics['eval/episode_reward_std'])

      plt.xlim([0, self.train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([self.min_y, self.max_y])

      plt.xlabel('# environment steps')
      plt.ylabel('reward per episode')
      plt.title(f'y={self.y_data[-1]:.3f}')

      plt.errorbar(
          self.x_data, self.y_data, yerr=self.ydataerr)
      plt.show()

      cwd = os.path.dirname(os.path.abspath(__file__))
      os.makedirs(cwd + '/myosuite-mjx-policies/', exist_ok=True)
      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{self.experiment_id}_progress.png')
      plt.savefig(fig_path)
    elif 'episode/sum_reward' in metrics:
      ## called during training
          # print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
          # task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
          # episode length: {metrics['eval/avg_episode_length']}")
      self.times.append(datetime.now())
      self.x_data_train.append(num_steps)
      self.y_data_train.append(metrics['episode/sum_reward'])
      self.y_data_train_length.append(metrics['episode/length'])
      self.y_data_train_success.append(metrics['episode/success_rate'])
      self.y_data_train_curriculum_state.append(metrics['episode/target_area_dynamic_width_scale'])

      cwd = os.path.dirname(os.path.abspath(__file__))
      os.makedirs(cwd + '/myosuite-mjx-policies/', exist_ok=True)

      ## Reward
      plt.xlim([0, self.train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([self.min_y, self.max_y])

      plt.xlabel('# environment steps')
      plt.ylabel('reward per episode')
      plt.title(f'y={self.y_data_train[-1]:.3f}')

      plt.errorbar(
          self.x_data_train, self.y_data_train) # yerr=self.ydataerr)
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{self.experiment_id}_progress_train_reward.png')
      plt.savefig(fig_path)
      plt.close()

      ## Episode length
      plt.xlim([0, self.train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([0, self.max_episode_length * 1.1])

      plt.xlabel('# environment steps')
      plt.ylabel('episode length')
      plt.title(f'length={self.y_data_train_length[-1]:.3f}')
      plt.errorbar(
          self.x_data_train, self.y_data_train_length) # yerr=ydataerr)
      plt.show()
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{self.experiment_id}_progress_train_length.png')
      plt.savefig(fig_path)
      plt.close()
      
      ## Success rate
      plt.xlim([0, self.train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([0, 1])

      plt.xlabel('# environment steps')
      plt.ylabel('success rate')
      plt.title(f'success={self.y_data_train_success[-1]:.3f}')
      plt.errorbar(
          self.x_data_train, self.y_data_train_success) # yerr=ydataerr)
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{self.experiment_id}_progress_train_success.png')
      plt.savefig(fig_path)
      plt.close()
      
      ## Curriculum state
      plt.xlim([0, self.train_fn.keywords['num_timesteps'] * 1.25])
      plt.ylim([0, 1])

      plt.xlabel('# environment steps')
      plt.ylabel('curriculum state')
      plt.title(f'width={self.y_data_train_curriculum_state[-1]:.3f}')
      plt.errorbar(
          self.x_data_train, self.y_data_train_curriculum_state) # yerr=ydataerr)
      plt.show()

      fig_path = os.path.join(cwd, f'myosuite-mjx-policies/{self.experiment_id}_progress_train_curriculum.png')
      try:
        plt.savefig(fig_path)
        plt.close()
      except:
        pass

# %%
experiment_id='ArmReach with checkpoint and init_target_area_width_scale=0.0'
n_train_steps=100_000_000
n_eval_eps=10
restore_params_path=None
init_target_area_width_scale=0.

env_name = 'mobl_arms_index_llc_eepos_adaptive_mjx-v0'
from myosuite.envs.myo.myouser.llc_eepos_adaptive_mjx_v2 import LLCEEPosAdaptiveEnvMJXV0, LLCEEPosAdaptiveDirectCtrlEnvMJXV0
# envs.register_environment(env_name, LLCEEPosAdaptiveEnvMJXV0)

model_path = 'simhive/uitb_sim/mobl_arms_index_eepos_pointing.xml'
path = (epath.Path(epath.resource_path('myosuite')) / (model_path)).as_posix()

kwargs = {
          'frame_skip': 25,
         'target_pos_range': {'fingertip': jp.array([[0.225, 0.02, -0.09], [0.35, 0.32, 0.25]]),},
          'target_radius_range': {'fingertip': jp.array([0.025, 0.025]),},
          'ref_site': 'humphant',
          'adaptive_task': True,
          'init_target_area_width_scale': 0.,

          # 'adaptive_increase_success_rate': 0.6,
          # 'adaptive_decrease_success_rate': 0.3,
          # 'adaptive_change_step_size': 0.05,
          # 'adaptive_change_min_trials': 50,
          'success_log_buffer_length': 500,
          # 'normalize_act': True,
          'reset_type': 'range_uniform',
          # 'max_trials': 10
          'num_envs': 4096, # down from 3072
          'vision': {
              'gpu_id': 0,
              'render_width': 120,
              'render_height': 120,
              'enabled_cameras': [0],
          }
      }

env = LLCEEPosAdaptiveEnvMJXV0(model_path=path, **kwargs)

restore_params = None

# %%
wrapped_env = wrap_for_brax_training(env, vision=True, num_vision_envs=kwargs['num_envs'])

# %%
def custom_network_factory(obs_shape, action_size, preprocess_observations_fn):
    return networks_vision.make_ppo_networks_vision(
        observation_size={
        "pixels/view_0": (120, 120, 3),  # RGB image
        "proprioception": (48,)          # Vector state
        }, 
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(512, 256, 64),  # Larger policy network
        value_hidden_layer_sizes=(256, 256, 64),   # Standard value network
        policy_obs_key="proprioception",
        value_obs_key="proprioception",
        normalise_channels=True            # Normalize image channels
    )

network_factory = custom_network_factory

# %%
train_fn = functools.partial(
    ppo.train, 
    num_timesteps=n_train_steps, # from n_train_steps,
    madrona_backend=True,
    num_evals=0, reward_scaling=0.1,
    wrap_env=False, episode_length=800, #when wrap_curriculum_training is used, 'episode_length' only determines length of eval episodes
    normalize_observations=True, action_repeat=1,
    unroll_length=10, 
    num_minibatches=16, #down from 24
    num_updates_per_batch=16, #up from 8
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, 
    num_envs=kwargs['num_envs'], #down from 3072
    num_eval_envs=kwargs['num_envs'], #must be same as num_envs
    batch_size=256, #down from 512
    seed=0,
    log_training_metrics=True,
    restore_params=restore_params, 
    network_factory=network_factory,
    save_checkpoint_path=f'myosuite-mjx-policies-checkpoint/{experiment_id}_params'
    )
## rule of thumb: num_timesteps ~= (unroll_length * batch_size * num_minibatches) * [desired number of policy updates (internal variabele: "num_training_steps_per_epoch")]; Note: for fixed total env steps (num_timesteps), num_evals and num_resets_per_eval define how often policies are evaluated and the env is reset during training (split into Brax training "epochs")

progress_cls = Progress(train_fn, experiment_id)

# %%
import wandb
wandb.init(project='myosuite-mjx-policies', name=experiment_id)
make_inference_fn, params, metrics = train_fn(environment=wrapped_env, progress_fn=progress_cls.progress)

# %%
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))


# %%
times = progress_cls.times

if n_train_steps > 0 and len(times) > 2:
    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

os.makedirs('myosuite-mjx-policies/', exist_ok=True)

param_path = os.path.join(f'myosuite-mjx-policies/{experiment_id}_params')
if n_train_steps > 0:
    model.save_params(param_path, params)


