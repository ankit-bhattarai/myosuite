import os
from xml.parsers.expat import model
import jax
from ml_collections import ConfigDict
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import jax.numpy as jnp

def evaluate_policy(checkpoint_path=None, env_name=None,
                    eval_env=None, jit_inference_fn=None, jit_reset=None, jit_step=None,
                    seed=123, n_episodes=1,
                    ep_length=None, keys=None):
    """
    Generate an evaluation trajectory from a stored checkpoint policy.

    You can either call this method by directly passing the checkpoint, env, jitted policy, etc. (useful if checkpoint was already loaded in advance, e.g., in a Jupyter notebook file), 
    or let this method load the env and policy from scratch by passing only checkpoint_path and env_name (takes ~1min).
    """

    print("start evaluate")

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

    for _ in range(n_episodes):
        state = jit_reset(reset_keys)
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

        #after one episode:
        L = abs(state.info["end_line"][1] - state.info["start_line"][1])
        W = abs(state.info["top_line"][2] - state.info["bottom_line"][2])
        ID = L/W
        IDs.append(ID)
        MTs.append(MT)
        print(ID, MT)
        #state.metrics.append({"ID": ID, "MT": MT})
        completed_phase_0 = False
        
    model = LinearRegression()
    model.fit(IDs, MTs)

    X = jnp.array(IDs).reshape(-1, 1)
    y = jnp.array(MTs)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    state.metrics.append({"R2": r2})

    return rollout