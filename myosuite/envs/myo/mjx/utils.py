import os
import functools
import json
from etils import epath
import jax

from mujoco_playground import registry, wrapper
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo

from myosuite.train.myouser.custom_ppo import train as ppo
from myosuite.train.myouser.custom_ppo import networks_vision_unified as networks
from myosuite.train.utils.wrapper import wrap_curriculum_training

def get_latest_run_path(base_path="logs/"):
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and os.path.isdir(os.path.join(base_path, d, "checkpoints")) and len(os.listdir(os.path.join(base_path, d, "checkpoints"))) > 1]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_path, d)))
    return os.path.join(base_path, latest_subdir, "checkpoints")

def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )

def load_checkpoint(env_name, 
                    env_cfg,
                    get_observation_size, #TODO: remove this dependency
                    eval_mode=False,
                    ppo_params=None,
                    logdir=None,
                    checkpoint_path=None,
                    policy_params_fn_checkpoints=None,
                    progress_fn=None,
                    vision=False,
                    domain_randomization=False,
                    rscope_envs=None,
                    deterministic_rscope=True,
                    seed=1,):
    
    # env_cfg = registry.get_default_config(env_name)  #default_config()
    ppo_params = ppo_params if ppo_params is not None else env_cfg.ppo_config
    if eval_mode:
        ppo_params.num_timesteps = 0  #only load the model, do not train

    # Handle checkpoint loading
    if checkpoint_path is not None:
        # Convert to absolute path
        ckpt_path = epath.Path(checkpoint_path).resolve()
        if ckpt_path.is_dir():
            latest_ckpts = list(ckpt_path.glob("*"))
            latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
            latest_ckpts.sort(key=lambda x: int(x.name))
            latest_ckpt = latest_ckpts[-1]
            restore_checkpoint_path = latest_ckpt
            print(f"Restoring from: {restore_checkpoint_path}")
        else:
            restore_checkpoint_path = ckpt_path
            print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    else:
        print("No checkpoint path provided, not restoring from checkpoint")
        restore_checkpoint_path = None

    # Set up checkpoint directory
    if logdir is not None:
        ckpt_path = logdir / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint path: {ckpt_path}")
    else:
        assert checkpoint_path is not None, "Either logdir or checkpoint_path must be provided"
        print(f"Checkpoint path: {ckpt_path}")

    # Save environment configuration
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict(), fp, indent=4)

    if vision:
        env_cfg.vision = True
        env_cfg.vision_config.render_batch_size = ppo_params.num_envs
    env = registry.load(env_name, config=env_cfg)

    if policy_params_fn_checkpoints is None:
        from orbax import checkpoint as ocp
        from flax.training import orbax_utils

        # Define policy parameters function for saving checkpoints
        def policy_params_fn_checkpoints(current_step, make_policy, params):
            """Function to save policy parameters."""
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params)
            path = ckpt_path / f"{current_step}"
            orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    # network_fn = (
    #     ppo_networks_vision.make_ppo_networks_vision
    #     if vision
    #     else ppo_networks.make_ppo_networks
    # )
    # if hasattr(ppo_params, "network_factory"):
    #     network_factory = functools.partial(
    #         network_fn, **ppo_params.network_factory
    #     )
    # else:
    #     network_factory = network_fn

    network_factory = functools.partial(networks.custom_network_factory, vision=vision, get_observation_size=functools.partial(get_observation_size, vision=vision), **getattr(ppo_params, "network_factory", {}))

    if domain_randomization:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            env_name
        )

    if vision:
        env = wrap_curriculum_training(
            env,
            vision=True,
            num_vision_envs=env_cfg.vision_config.render_batch_size,
            episode_length=ppo_params.episode_length,
            action_repeat=ppo_params.action_repeat,
            randomization_fn=training_params.get("randomization_fn"),
        )

    num_eval_envs = (
        ppo_params.num_envs
        if vision
        else ppo_params.get("num_eval_envs", 128)
    )

    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        # policy_params_fn=policy_params_fn,
        seed=seed,
        restore_checkpoint_path=restore_checkpoint_path,
        # save_checkpoint_path=ckpt_path,
        # wrap_env_fn=None if vision else wrapper.wrap_for_brax_training,
        wrap_env_fn=None if vision else wrap_curriculum_training,
        num_eval_envs=num_eval_envs,
    )

    # Load evaluation environment
    eval_env = (
        None if vision else registry.load(env_name, config=env_cfg)
    )

    if rscope_envs:
        # Interactive visualisation of policy checkpoints
        from rscope import brax as rscope_utils

        if not vision:
            rscope_env = registry.load(env_name, config=env_cfg)
            rscope_env = wrap_curriculum_training(  #wrapper.wrap_for_brax_training(
                rscope_env,
                episode_length=ppo_params.episode_length,
                action_repeat=ppo_params.action_repeat,
                randomization_fn=training_params.get("randomization_fn"),
            )
        else:
            rscope_env = env

        rscope_handle = rscope_utils.BraxRolloutSaver(
            rscope_env,
            ppo_params,
            vision,
            rscope_env,
            deterministic_rscope,
            jax.random.PRNGKey(seed),
            rscope_fn,
        )

        def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
            policy_params_fn_checkpoints(current_step, make_policy, params)
            rscope_handle.set_make_policy(make_policy)
            rscope_handle.dump_rollout(params)
    else:
        policy_params_fn = policy_params_fn_checkpoints
    
    if not eval_mode:
        print("Starting to JIT compile...")
        
    # Train or load the model
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,  #lambda *args: None,
        eval_env=None if vision else eval_env,
    )
    if vision:
        eval_env = env
    eval_env.enable_eval_mode()

    return eval_env, make_inference_fn, params

