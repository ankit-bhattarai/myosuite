import os
from typing import Callable, List, Optional, Sequence
import mujoco
from mujoco_playground import State
from mujoco_playground._src import mjx_env
import numpy as np

import jax
from myosuite.train.utils.train import train_or_load_checkpoint
from ml_collections import ConfigDict
import json
import mediapy as media
import matplotlib.pyplot as plt
import wandb
from datetime import datetime


def evaluate_policy(checkpoint_path=None, env_name=None,
                    eval_env=None, env_cfg=None, jit_inference_fn=None, jit_reset=None, jit_step=None,
                    seed=123, n_episodes=1):
    """
    Generate an evaluation trajectory from a stored checkpoint policy.

    You can either call this method by directly passing the checkpoint, env, jitted policy, etc. (useful if checkpoint was already loaded in advance, e.g., in a Jupyter notebook file), 
    or let this method load the env and policy from scratch by passing only checkpoint_path and env_name (takes ~1min).
    """

    if checkpoint_path is None:
        assert eval_env is not None, "If no checkpoint path is provided, env must be passed directly as 'eval_env'"
        ## TODO: directly pass episode_length rather than env_cfg, if checkpoint_path is None
        assert env_cfg is not None, "If no checkpoint path is provided, env config must be passed directly as 'env_cfg'"
        assert jit_inference_fn is not None, "If no checkpoint path is provided, policy must be passed directly as 'jit_inference_fn'"
        assert jit_reset is not None, "If no checkpoint path is provided, jitted reset function must be passed directly as 'jit_reset'"
        assert jit_step is not None, "If no checkpoint path is provided, jitted step function must be passed directly as 'jit_step'"
    else:
        assert env_name is not None, "If checkpoint path is provided, env name must also be passed as 'env_name'"
        with open(os.path.join(checkpoint_path, "config.json"), "r") as f:
            env_cfg = ConfigDict(json.load(f))
        eval_env, make_inference_fn, params = train_or_load_checkpoint(env_name, env_cfg, ppo_params=env_cfg.ppo_config, eval_mode=True, checkpoint_path=checkpoint_path)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)

    eval_key = jax.random.PRNGKey(seed)
    eval_key, reset_keys = jax.random.split(eval_key)
    rollout = []
    # modify_scene_fns = []

    for _ in range(n_episodes):
        state = jit_reset(reset_keys)
        rollout.append(state)
        # modify_scene_fns.append(functools.partial(update_target_visuals, target_pos=state.info["target_pos"].flatten(), target_size=state.info["target_radius"].flatten()))
        for i in range(env_cfg.ppo_config.episode_length):
            eval_key, key = jax.random.split(eval_key)
            ctrl, _ = jit_inference_fn(state.obs, key)  #VARIANT 1
            # ctrl = deterministic_policy(state.obs)  #VARIANT 2
            # ctrl = jax.random.uniform(act_rng, shape=eval_env._na)  #BASELINE: random control
            state = jit_step(state, ctrl)
            # touch_detected = any([({eval_env.mj_model.geom(con_geom[0]).name, eval_env.mj_model.geom(con_geom[1]).name} == {"fingertip_contact", "screen"}) and (con_dist < 0) for con_geom, con_dist in zip(state.data._impl.contact.geom, state.data._impl.contact.dist)])
            # if touch_detected:
            #   print(f"Step {i}, touch detected. {state.data._impl.contact.dist}")    
            # print(f"Step {i}, ee_pos: {state.obs[:, 15:18]}")
            # print(f"Target {i}, target_pos: {state.obs[:, -4:-1]}")
            # print(f"Step {i}, steps_since_last_hit: {state.info['steps_since_last_hit']}")
            rollout.append(state)
            # modify_scene_fns.append(functools.partial(update_target_visuals, target_pos=state.info["target_pos"].flatten(), target_size=state.info["target_radius"].flatten()))
            if state.done.all():
                break
        eval_key, reset_keys = jax.random.split(eval_key)

    return rollout

def render_traj(rollout: List[State],
                eval_env: mjx_env.MjxEnv, 
                height: int = 240,
                width: int = 320,
                render_every: int = 1, 
                camera: Optional[int | str] = "fixed-eye",
                scene_option: Optional[mujoco.MjvOption] = None,
                modify_scene_fns: Optional[
                    Sequence[Callable[[mujoco.MjvScene], None]]
                ] = None,
                notebook_context: bool = True):
    ## if called outside a jupyter notebook file, use notebook_context=False

    traj = rollout[::render_every]
    if modify_scene_fns is not None:
        modify_scene_fns = modify_scene_fns[::render_every]
    frames = eval_env.render(traj, height=height, width=width, camera=camera,
                            scene_option=scene_option, modify_scene_fns=modify_scene_fns)
    # rewards = [s.reward for s in rollout]
    
    if notebook_context:
        media.show_video(frames, fps=1.0 / eval_env.dt / render_every)
    else:
        #return media.show_video(frames, fps=1.0 / eval_env.dt / render_every, return_html=True)
        return frames

def update_target_visuals(scn, target_pos, target_size,
                          rgba=[0., 1., 0., 1.]):
    """Updates newly created target visuals in the scene.
    Note: Requires to hide any target visuals included in the MjModel XML."""

    assert scn.ngeom < scn.maxgeom, "Too many geoms in the scene"
    scn.ngeom += 1

    # initialise a new sphere
    mujoco.mjv_initGeom(
          scn.geoms[scn.ngeom-1],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[target_size.item(), 0, 0],
          pos=target_pos,
          mat=np.eye(3).flatten(),
          rgba=np.array(rgba).astype(np.float32)
      )
    # mujoco.mjv_connector(scn.geoms[scn.ngeom-1],
    #                     mujoco.mjtGeom.mjGEOM_CAPSULE, target_size.item(),
    #                     target_pos, target_pos + np.array([1e-6, 0, 0]))

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

def set_global_seed(seed=0):
    """Set global random seeds for reproducible results."""
    import random
    import numpy as np
        
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    print(f"Global random seed set to {seed} for reproducible results")