import os
import functools
from etils import epath

from mujoco_playground import registry, wrapper
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo

def get_latest_run_path(base_path="logs/"):
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and os.path.isdir(os.path.join(base_path, d, "checkpoints")) and len(os.listdir(os.path.join(base_path, d, "checkpoints"))) > 1]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_path, d)))
    return os.path.join(base_path, latest_subdir, "checkpoints")

def load_checkpoint(env_name, checkpoint_path,
                    ppo_params=None,
                    vision=False,
                    domain_randomization=False,
                    seed=1,):
    env_cfg = registry.get_default_config(env_name)  #default_config()

    ppo_params = env_cfg.ppo_config
    ppo_params.num_timesteps = 0  #only load the model, do not train

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

    if vision:
        env_cfg.vision = True
        env_cfg.vision_config.render_batch_size = ppo_params.num_envs
    env = registry.load(env_name, config=env_cfg)

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]

    network_fn = (
        ppo_networks_vision.make_ppo_networks_vision
        if vision
        else ppo_networks.make_ppo_networks
    )
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(
            network_fn, **ppo_params.network_factory
        )
    else:
        network_factory = network_fn

    if domain_randomization:
        training_params["randomization_fn"] = registry.get_domain_randomizer(
            env_name
        )

    if vision:
        env = wrapper.wrap_for_brax_training(
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
        save_checkpoint_path=ckpt_path,
        wrap_env_fn=None if vision else wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )

    # Load evaluation environment
    eval_env = (
        None if vision else registry.load(env_name, config=env_cfg)
    )
        
    # Train or load the model
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        # progress_fn=progress,
        policy_params_fn=lambda *args: None,
        eval_env=None if vision else eval_env,
    )
    if vision:
        eval_env = env

    return eval_env, make_inference_fn, params

