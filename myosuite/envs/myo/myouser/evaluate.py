import os
import jax
from ml_collections import ConfigDict
import json


def evaluate_policy(checkpoint_path=None, env_name=None,
                    eval_env=None, jit_inference_fn=None, jit_reset=None, jit_step=None,
                    seed=123, n_episodes=1,
                    ep_length=None):
    """
    Generate an evaluation trajectory from a stored checkpoint policy.

    You can either call this method by directly passing the checkpoint, env, jitted policy, etc. (useful if checkpoint was already loaded in advance, e.g., in a Jupyter notebook file), 
    or let this method load the env and policy from scratch by passing only checkpoint_path and env_name (takes ~1min).
    """

    if checkpoint_path is None:
        assert eval_env is not None, "If no checkpoint path is provided, env must be passed directly as 'eval_env'"
        assert jit_inference_fn is not None, "If no checkpoint path is provided, policy must be passed directly as 'jit_inference_fn'"
        assert jit_reset is not None, "If no checkpoint path is provided, jitted reset function must be passed directly as 'jit_reset'"
        assert jit_step is not None, "If no checkpoint path is provided, jitted step function must be passed directly as 'jit_step'"
    else:
        assert env_name is not None, "If checkpoint path is provided, env name must also be passed as 'env_name'"
        from myosuite.train.utils.train import train_or_load_checkpoint
        with open(os.path.join(checkpoint_path, "config.json"), "r") as f:
            config = ConfigDict(json.load(f))
        eval_env, make_inference_fn, params = train_or_load_checkpoint(env_name, config, eval_mode=True, checkpoint_path=checkpoint_path)
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)

    eval_key = jax.random.PRNGKey(seed)
    eval_key, reset_keys = jax.random.split(eval_key)
    rollout = []
    # modify_scene_fns = []

    if ep_length is None:
        ep_length = int(eval_env._config.task_config.max_duration / eval_env._config.ctrl_dt)  #TODO

    for _ in range(n_episodes):
        state = jit_reset(reset_keys)
        rollout.append(state)
        # modify_scene_fns.append(functools.partial(update_target_visuals, target_pos=state.info["target_pos"].flatten(), target_size=state.info["target_radius"].flatten()))
        for i in range(ep_length):
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