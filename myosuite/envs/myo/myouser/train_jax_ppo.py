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
"""Train a PPO agent using JAX on the specified environment."""

from datetime import datetime
import os
import warnings
import pickle
import h5py

import jax

from absl import app
from absl import flags
from absl import logging
from myosuite.train.utils.train import train_or_load_checkpoint
from myosuite.envs.myo.myouser.evaluate import evaluate_policy
from myosuite.envs.myo.myouser.utils import render_traj, ProgressLogger, set_global_seed
from etils import epath
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media

from myosuite.envs.myo.myouser.myouser_steering_v0 import calculate_metrics

from tensorboardX import SummaryWriter
import wandb

from mujoco_playground import registry


import hydra
from hydra_cli import Config
from omegaconf import OmegaConf
from ml_collections.config_dict import ConfigDict


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
try:
  import madrona_mjx
  location = madrona_mjx.__file__.split('/src/')[0]
  os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = f"{location}/build/kernel_cache"
  os.environ["MADRONA_BVH_KERNEL_CACHE"] = f"{location}/build/bvh_cache"
  print(f'Using cached Madrona MJX kernels at: {location}/build')
except:
  print("Madrona MJX not found, can't use any vision components for training!")

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")



@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
  container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
  container['env']['vision'] = container['vision']
  config = ConfigDict(container)
  print(f"Config: ", OmegaConf.to_yaml(container), sep='\n')
  """Run training and evaluation for the specified environment."""

  # Set global seed for reproducibility
  set_global_seed(config.run.seed)

  print(f"Current backend: {jax.default_backend()}")
  # Load environment configuration
  env_cfg = config.env

  ppo_params = config.rl

  # Generate unique experiment name
  if not config.wandb.enabled:  
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_cfg.env_name}-{timestamp}"
    suffix = "" if config.run.suffix is None else f"-{config.run.suffix}"
    exp_name += suffix
  else:
    exp_name = config.wandb.name
  print(f"Experiment name: {exp_name}")
  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if config.wandb.enabled and not config.run.play_only:
    wandb_params = config.wandb.to_dict()
    wandb_params.pop('enabled')
    wandb.init(**wandb_params)
    wandb.config.update(config.to_dict())
    wandb.config.update({"env_name": env_cfg.env_name})

  # Initialize TensorBoard if required
  if config.run.use_tb and not config.run.play_only:
    writer = SummaryWriter(logdir)
  else:
    writer = None
  
  progress_logger = ProgressLogger(writer=writer, ppo_params=ppo_params, logdir=logdir,
                                   local_plotting=config.run.local_plotting, log_wandb=config.wandb.enabled, log_tb=config.run.use_tb)
  progress_fn = progress_logger.progress

  ## TRAINING/LOADING CHECKPOINT
  # Train or load the model
  env, make_inference_fn, params = train_or_load_checkpoint(env_cfg.env_name, config,
                    logdir=logdir,
                    checkpoint_path=config.rl.load_checkpoint_path,
                    progress_fn=progress_fn,
                    log_wandb_videos=config.wandb.enabled and not config.run.play_only,
                    vision=config.vision.enabled,
                    domain_randomization=config.run.domain_randomization,
                    rscope_envs=config.run.rscope_envs,
                    deterministic_rscope=config.run.deterministic_rscope,
                    seed=config.run.seed)
  if ppo_params.num_timesteps > 0:
    print("Done training.")
  else:
    print("Done loading checkpoint.")

  times = progress_logger.times
  if ppo_params.num_timesteps > 0 and len(times) > 2:
    # print(f'time to JIT compile: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

  ## STORING CHECKPOINT
  with open(logdir / 'playground_params.pickle', 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  ## EVALUATING CHECKPOINT
  print("Starting inference...")

  # Jit required functions
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)

  if env_cfg.env_name=="MyoUserSteering":
    n_episodes = 50
  else:
    n_episodes = 1

  # Prepare for evaluation
  rollout = evaluate_policy(#checkpoint_path=_LOAD_CHECKPOINT_PATH.value, env_name=_ENV_NAME.value,
                            eval_env=env, jit_inference_fn=jit_inference_fn, jit_reset=jit_reset, jit_step=jit_step,
                            seed=123,  #seed=_SEED.value,  #TODO: add eval_seed to hydra/config
                            n_episodes=n_episodes)  #TODO: n_episodes as hydra config param?
  print(f"Return: {jp.array([r.reward for r in rollout]).sum()}")
  print(f"env: {env_cfg.env_name}")
  if env_cfg.env_name=="MyoUserSteering":
    metrics = calculate_metrics(rollout, ['R^2'])
    wandb.log(metrics)

  # Render and save the rollout
  render_every = 2
  fps = 1.0 / env.dt / render_every
  print(f"FPS for rendering: {fps}")
  traj = rollout[::render_every]
  with h5py.File(logdir / 'traj.h5', 'w') as h5f:
      h5f.create_dataset('qpos', data=[s.data.qpos for s in traj])
      h5f.create_dataset('ctrl', data=[s.data.ctrl for s in traj])
      h5f.close()
  with open(logdir / 'traj.pickle', 'wb') as handle:
      pickle.dump(traj, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  # scene_option = mujoco.MjvOption()
  # scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  # scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  # render front view
  frames = render_traj(
      traj, env, height=480, width=640, camera="fixed-eye",
      notebook_context=False,
      #scene_option=scene_option
  )
  media.write_video(logdir / "rollout.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout.mp4'.")
  if config.wandb.enabled and not config.run.play_only:
    wandb.log({'final_policy/front_view': wandb.Video(str(logdir / "rollout.mp4"), format="mp4")})  #, fps=fps)})

  # render side view
  frames = render_traj(
      traj, env, height=480, width=640, camera=None,
      notebook_context=False,
      #scene_option=scene_option
  )
  media.write_video(logdir / "rollout_1.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout_1.mp4'.")
  if config.wandb.enabled and not config.run.play_only:
    wandb.log({'final_policy/side_view': wandb.Video(str(logdir / "rollout_1.mp4"), format="mp4")})  #, fps=fps)})


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()  #allow for debugging flags such as --jax_debug_nans=True or --jax_disable_jit=True
  main()