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
from myosuite.envs.myo.myouser.utils import evaluate_policy, render_traj
from etils import epath
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media

from tensorboardX import SummaryWriter
import wandb

from mujoco_playground import registry
# from mujoco_playground.config import dm_control_suite_params
# from mujoco_playground.config import locomotion_params
# from mujoco_playground.config import manipulation_params

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
try:
  import madrona_mjx
  location = madrona_mjx.__file__.split('/src/')[0]
  os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = f"{location}/build/kernel_cache"
  os.environ["MADRONA_BVH_KERNEL_CACHE"] = f"{location}/build/bvh_cache"
  print(f'Using cached Madrona MJX kernels at: {location}/build')
except:
  print("Madrona MJX not found, can't use any vision components for training!")

os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "MyoElbow",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_string("vision", '', "Whether and which type of vision input to use")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    True,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", True, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    True,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)

class ProgressLogger:
  def __init__(self, writer=None, ppo_params=None, local_plotting=False, logdir=None, log_wandb=True, log_tb=True):
    self.times = [datetime.now()]
    self.writer = writer
    self.ppo_params = ppo_params
    self.local_plotting = local_plotting
    self.log_wandb = log_wandb
    self.log_tb = log_tb

    if self.local_plotting:
      assert logdir is not None, "logdir must be provided if local_plotting is True"
      self.logdir = logdir

      class PlotParams(object):
        def __init__(self, ppo_params):
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
          self.max_episode_length = ppo_params.episode_length
      self._plt_params = PlotParams(self.ppo_params)
  
  def progress(self, num_steps, metrics):
    self.times.append(datetime.now())
    if len(self.times) == 2:
        print(f'time to JIT compile: {self.times[1] - self.times[0]}')
        print(f'Starting training...')

    # Log to Weights & Biases
    if self.log_wandb:
      wandb.log({'num_steps': num_steps, **metrics}, step=num_steps)

    # Log to TensorBoard
    if self.log_tb and self.writer is not None:
      for key, value in metrics.items():
        self.writer.add_scalar(key, value, num_steps)
      self.writer.flush()

    if 'eval/episode_reward' in metrics:
    # if self.ppo_params.num_evals > 0:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f} episode_length={metrics['eval/avg_episode_length']:.3f}")
    if not self.log_wandb and not self.log_tb and self.ppo_params.log_training_metrics:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )
    
    if self.local_plotting:
      if self.ppo_params.num_evals > 0 and 'eval/episode_reward' in metrics:
        ## called during evaluation
        # print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
        #     task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
        #     episode length: {metrics['eval/avg_episode_length']}")
        self._plt_params.times.append(datetime.now())
        self._plt_params.x_data.append(num_steps)
        self._plt_params.y_data.append(metrics['eval/episode_reward'])
        self._plt_params.ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([self._plt_params.min_y, self._plt_params.max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={self._plt_params.y_data[-1]:.3f}')

        plt.errorbar(
            self._plt_params.x_data, self._plt_params.y_data, yerr=self._plt_params.ydataerr)
        plt.show()
        
        fig_path = self.logdir / 'progress.png'
        plt.savefig(fig_path)
      elif self.ppo_params.log_training_metrics and 'episode/sum_reward' in metrics:
        ## called during training
            # print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}, \
            # task coverage: {metrics['eval/episode_target_area_dynamic_width_scale']}, success rate: {metrics['eval/episode_success_rate']}, \
            # episode length: {metrics['eval/avg_episode_length']}")
        self._plt_params.times.append(datetime.now())
        self._plt_params.x_data_train.append(num_steps)
        self._plt_params.y_data_train.append(metrics['episode/sum_reward'])
        self._plt_params.y_data_train_length.append(metrics['episode/length'])
        self._plt_params.y_data_train_success.append(metrics['episode/success_rate'])
        self._plt_params.y_data_train_curriculum_state.append(metrics['episode/target_area_dynamic_width_scale'])

        ## Reward
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([self._plt_params.min_y, self._plt_params.max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={self._plt_params.y_data_train[-1]:.3f}')

        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train) # yerr=ydataerr)
        plt.show()

        fig_path = self.logdir / 'progress_train_reward.png'
        plt.savefig(fig_path)
        plt.close()

        ## Episode length
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([0, self._plt_params.max_episode_length * 1.1])

        plt.xlabel('# environment steps')
        plt.ylabel('episode length')
        plt.title(f'length={self._plt_params.y_data_train_length[-1]:.3f}')
        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train_length) # yerr=ydataerr)
        plt.show()
        plt.show()

        fig_path = self.logdir / 'progress_train_length.png'
        plt.savefig(fig_path)
        plt.close()
        
        ## Success rate
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([0, 1])

        plt.xlabel('# environment steps')
        plt.ylabel('success rate')
        plt.title(f'success={self._plt_params.y_data_train_success[-1]:.3f}')
        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train_success) # yerr=ydataerr)
        plt.show()

        fig_path = self.logdir / 'progress_train_success.png'
        plt.savefig(fig_path)
        plt.close()
        
        ## Curriculum state
        plt.xlim([0, self.ppo_params.num_timesteps * 1.25])
        plt.ylim([0, 1])

        plt.xlabel('# environment steps')
        plt.ylabel('curriculum state')
        plt.title(f'width={self._plt_params.y_data_train_curriculum_state[-1]:.3f}')
        plt.errorbar(
            self._plt_params.x_data_train, self._plt_params.y_data_train_curriculum_state) # yerr=ydataerr)
        plt.show()

        fig_path = self.logdir / 'progress_train_curriculum.png'
        try:
          plt.savefig(fig_path)
          plt.close()
        except:
          pass

class ProgressEvalVideoLogger:
  def __init__(self, logdir, eval_env, log_wandb=True):
    self.logdir = logdir
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    self.checkpoint_path = ckpt_path
    self.eval_env = eval_env

  def progress_eval_video(self, num_steps, make_policy, params):
    # Log to Weights & Biases
    if self.log_wandb:
      # Rollout trajectory
      ## TODO: rollout policy (see e.g. rscope.BraxRolloutSaver class for reference)
      raise NotImplementedError()
      # make_policy(params, deterministic=True)
      # rollout = evaluate_policy(checkpoint_path=self.checkpoint_path, env_name=env_name)
      
      fps = 1.0 / self.eval_env.dt

      # render front view
      frames = render_traj(
          rollout, self.eval_env, height=480, width=640, camera="fixed-eye",
          notebook_context=False,
      )
      media.write_video(self.checkpoint_path / f"{num_steps}.mp4", frames, fps=fps)
      if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb.log({'eval/front_view': wandb.Video(str(self.checkpoint_path / f"{num_steps}.mp4"), format="mp4")}, step=num_steps)  #, fps=fps)}, step=num_steps)

      # render side view
      frames = render_traj(
          rollout, self.eval_env, height=480, width=640, camera=None,
          notebook_context=False,
      )
      media.write_video(self.checkpoint_path / f"{num_steps}_1.mp4", frames, fps=fps)
      if _USE_WANDB.value and not _PLAY_ONLY.value:
        wandb.log({'eval/side_view': wandb.Video(str(self.checkpoint_path / f"{num_steps}_1.mp4"), format="mp4")}, step=num_steps)  #, fps=fps)}, step=num_steps)

def set_global_seed(seed=0):
    """Set global random seeds for reproducible results."""
    import random
    import numpy as np
        
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    print(f"Global random seed set to {seed} for reproducible results")


# def get_rl_config(env_name: str) -> config_dict.ConfigDict:
#   if env_name in mujoco_playground.manipulation._envs:
#     if _VISION.value:
#       return manipulation_params.brax_vision_ppo_config(env_name)
#     return manipulation_params.brax_ppo_config(env_name)
#   elif env_name in mujoco_playground.locomotion._envs:
#     if _VISION.value:
#       return locomotion_params.brax_vision_ppo_config(env_name)
#     return locomotion_params.brax_ppo_config(env_name)
#   elif env_name in mujoco_playground.dm_control_suite._envs:
#     if _VISION.value:
#       return dm_control_suite_params.brax_vision_ppo_config(env_name)
#     return dm_control_suite_params.brax_ppo_config(env_name)

#   raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")

import hydra
from hydra_cli import Config
from omegaconf import OmegaConf
from ml_collections.config_dict import ConfigDict

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
  container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
  container['env']['vision'] = container['vision']
  config = ConfigDict(container)
  """Run training and evaluation for the specified environment."""

  # Set global seed for reproducibility
  set_global_seed(config.run.seed)

  print(f"Current backend: {jax.default_backend()}")
  # Load environment configuration
  env_cfg = config.env

  ppo_params = config.rl  #get_rl_config(_ENV_NAME.value)

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")


  # Generate unique experiment name
  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{env_cfg.env_name}-{timestamp}"
  if config.wandb.suffix is not None:
    exp_name += f"-{config.wandb.suffix}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if config.wandb.enabled and not config.run.play_only:
    wandb.init(project=config.wandb.project, name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": env_cfg.env_name})

  # Initialize TensorBoard if required
  if config.run.use_tb and not config.run.play_only:
    writer = SummaryWriter(logdir)
  else:
    writer = None
  
  progress_logger = ProgressLogger(writer=writer, ppo_params=ppo_params, logdir=logdir,
                                   local_plotting=False)
  progress_fn = progress_logger.progress

  # progress_eval_video_logger = ProgressEvalVideoLogger(logdir=logdir, eval_env=eval_env)
  # progress_fn_eval_video = progress_eval_video_logger.progress_eval_video
  progress_fn_eval_video = lambda *args: None

  # Train or load the model
  env, make_inference_fn, params = train_or_load_checkpoint(env_cfg.env_name, env_cfg,
                    ppo_params=ppo_params,
                    logdir=logdir,
                    checkpoint_path=config.rl.load_checkpoint_path,
                    progress_fn=progress_fn,
                    progress_fn_eval_video=progress_fn_eval_video,
                    vision=config.vision.enabled,
                    domain_randomization=config.run.domain_randomization,
                    rscope_envs=config.run.rscope_envs,
                    deterministic_rscope=config.run.deterministic_rscope,
                    seed=config.run.seed)

  # print("Done training.")
  # if len(times) > 1:
  #   print(f"Time to JIT compile: {times[1] - times[0]}")
  #   print(f"Time to train: {times[-1] - times[1]}")
  times = progress_logger.times
  if ppo_params.num_timesteps > 0 and len(times) > 2:
    # print(f'time to JIT compile: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

  with open(logdir / 'playground_params.pickle', 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print("Starting inference...")

  # Create inference function
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Prepare for evaluation
  ## TODO: use myosuite.envs.myo.myouser.utils.evaluate_policy instead
  breakpoint()
  eval_env = (
      None if config.vision.enabled else registry.load(env_cfg.env_name, config=env_cfg)
  )
  num_envs = 1
  breakpoint()
  if config.vision.enabled:
    eval_env = env
    num_envs = config.vision.num_worlds
  eval_env.enable_eval_mode()

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(123)
  rng, reset_rng = jax.random.split(rng)
  if config.vision.enabled:
    reset_rng = jp.asarray(jax.random.split(reset_rng, num_envs))
  state = jit_reset(reset_rng)
  state0 = (
      jax.tree_util.tree_map(lambda x: x[0], state) if config.vision.enabled else state
  )
  rollout = [state0]

  # Run evaluation rollout
  for _ in range(ppo_params.episode_length): #TODO: fix where episode_length should be defined
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    state0 = (
        jax.tree_util.tree_map(lambda x: x[0], state)
        if config.vision.enabled
        else state
    )
    rollout.append(state0)
    if state0.done:
      break
  print(f"Return: {jp.array([r.reward for r in rollout]).sum()}")

  # Render and save the rollout
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
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
      traj, eval_env, height=480, width=640, camera="fixed-eye",
      notebook_context=False,
      #scene_option=scene_option
  )
  media.write_video(logdir / "rollout.mp4", frames, fps=fps)
  print("Rollout video saved as 'rollout.mp4'.")
  if config.wandb.enabled and not config.run.play_only:
    wandb.log({'final_policy/front_view': wandb.Video(str(logdir / "rollout.mp4"), format="mp4")})  #, fps=fps)})

  # render side view
  frames = render_traj(
      traj, eval_env, height=480, width=640, camera=None,
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