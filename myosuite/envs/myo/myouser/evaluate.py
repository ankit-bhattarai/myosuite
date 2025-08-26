import os
from xml.parsers.expat import model
import jax
from ml_collections import ConfigDict
import json
import jax.numpy as jnp
import random

def evaluate_non_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed=123, n_episodes=1, ep_length=None, reset_info_kwargs={}):
    eval_key = jax.random.PRNGKey(seed)
    eval_key, reset_keys = jax.random.split(eval_key)
    state = jit_reset(jax.random.split(reset_keys, n_episodes), **reset_info_kwargs)
    unvmap = lambda x, i : jax.tree.map(lambda x: x[i], x)
    dones = jp.zeros(n_episodes)
    rollouts = {i: [] for i in range(n_episodes)}
    for i in range(ep_length):
        eval_key, key = jax.random.split(eval_key)
        ctrl, _ = jit_inference_fn(state.obs, jax.random.split(key, n_episodes))
        state = jit_step(state, ctrl)
        for i in range(n_episodes):
            if not dones[i]:
                stated_unvampped = unvmap(state, i)
                rollouts[i].append(stated_unvampped)
        dones = jp.logical_or(dones, state.done)
        if dones.all():
            break
    rollout = []
    for i in range(n_episodes):
        rollout.extend(rollouts[i])
    return rollout, "rollout"

def evaluate_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed=123, n_episodes=1, ep_length=None):
    eval_key = jax.random.PRNGKey(seed)
    eval_key, reset_keys = jax.random.split(eval_key)
    num_worlds = eval_env._config.vision.num_worlds
    state = jit_reset(jax.random.split(reset_keys, num_worlds))
    pixel_key = [key for key in state.obs.keys() if 'pixels' in key]
    assert len(pixel_key) == 1, "Only one pixel key is supported"
    pixel_key = pixel_key[0]
    unvmap_upto = lambda x, i : jax.tree.map(lambda x: x[:i], x)
    unvmap = lambda x, i : jax.tree.map(lambda x: x[i], x)
    extract_states = unvmap_upto(state, n_episodes)
    dones = jp.zeros(n_episodes)
    videos = {i: [] for i in range(n_episodes)}
    rollouts = {i: [] for i in range(n_episodes)}
    for i in range(ep_length):
        eval_key, key = jax.random.split(eval_key)
        ctrl, _ = jit_inference_fn(state.obs, jax.random.split(key, num_worlds))
        state = jit_step(state, ctrl)
        extract_states = unvmap_upto(state, n_episodes)
        for i in range(n_episodes):
            if not dones[i]:
                videos[i].append(extract_states.obs[pixel_key][i])
                stated_unvampped = unvmap(state, i)
                rollouts[i].append(stated_unvampped)
        dones = jp.logical_or(dones, extract_states.done)
        if dones.all():
            break
    videos_all = []
    rollout = []
    for i in range(n_episodes):
        videos_all.extend(videos[i])
        rollout.extend(rollouts[i])
    return (rollout, videos_all), "videos"

def random_positions_deprecated(L, W):
    pos_min, pos_max = -0.2, 0.3
    left = random.uniform(pos_min, pos_max - L)
    right = left + L
    bottom = random.uniform(pos_min, pos_max - W)
    top = bottom + W
    return left, right, bottom, top

def random_positions(L, W, right):
    pos_min, pos_max = -0.2, 0.3
    left = right - L
    bottom = random.uniform(pos_min, pos_max - W)
    top = bottom + W
    return left, bottom, top

def get_custom_tunnels(rng: jax.Array, screen_pos: jax.Array) -> dict[str, jax.Array]:
    tunnel_positions_total  = []
    IDs = [1, 2, 3, 4, 5]
    L_min, L_max = 0.05, 0.5
    W_min, W_max = 0.07, 0.6
    right = 0.3
    for ID in IDs:
        combos = 0
        while combos < 1:
            W = random.uniform(W_min, W_max)
            L = ID * W
            if L_min <= L <= L_max:
                tunnel_positions = {}
                left, bottom, top = random_positions(L, W, right)
                width_midway = (left + right) / 2
                height_midway = (top + bottom) / 2
                tunnel_positions['bottom_line'] = screen_pos + jnp.array([0., width_midway, bottom])
                tunnel_positions['top_line'] = screen_pos + jnp.array([0., width_midway, top])
                tunnel_positions['start_line'] = screen_pos + jnp.array([0., right, height_midway])
                tunnel_positions['end_line'] = screen_pos + jnp.array([0., left, height_midway])
                tunnel_positions['screen_pos'] = screen_pos
                combos += 1

                for i in range(3):
                    tunnel_positions_total.append(tunnel_positions)
    return tunnel_positions_total

def evaluate_policy(checkpoint_path=None, env_name=None,
                    eval_env=None, jit_inference_fn=None, jit_reset=None, jit_step=None,
                    seed=123, n_episodes=1,
                    ep_length=None, keys=None):
    """
    Generate an evaluation trajectory from a stored checkpoint policy.

    You can either call this method by directly passing the checkpoint, env, jitted policy, etc. (useful if checkpoint was already loaded in advance, e.g., in a Jupyter notebook file), 
    or let this method load the env and policy from scratch by passing only checkpoint_path and env_name (takes ~1min).
    """
    rng = jax.random.PRNGKey(42)
    tunnel_positions = get_custom_tunnels(rng, screen_pos=jnp.array([0.532445, -0.27, 0.993]))

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

    if keys is not None:
        reset_keys = keys
        eval_key = keys

    if ep_length is None:
        ep_length = int(eval_env._config.task_config.max_duration / eval_env._config.ctrl_dt)

    for episode in range(len(tunnel_positions)):
        jax.debug.print("evaluate episode {}", episode)
        state = jit_reset(reset_keys, tunnel_positions=tunnel_positions[episode])
        #print(state.info)
        rollout.append(state)
        
        completed_phase_0 = False
        MT = 0
        for i in range(ep_length):
            eval_key, key = jax.random.split(eval_key)
            ctrl, _ = jit_inference_fn(state.obs, key) 
            state = jit_step(state, ctrl)

            metrics = state.metrics

            if metrics["completed_phase_0"] == True and not completed_phase_0:                    
                completed_phase_0 = True
                timestep_start_steering = i

            if completed_phase_0 and metrics["completed_phase_1"] == True:
                timestep_end_steering = i
                MT = (timestep_end_steering - timestep_start_steering)*0.002 * 25

            rollout.append(state)
            
            if state.done.all():
                break
        eval_key, reset_keys = jax.random.split(eval_key)
        #state.metrics.append({"ID": ID, "MT": MT})
        completed_phase_0 = False
    
    return rollout

#    if eval_env.vision:
#        return evaluate_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed, n_episodes, ep_length)
#    else:
#        return evaluate_non_vision(eval_env, jit_inference_fn, jit_reset, jit_step, seed, n_episodes, ep_length)
