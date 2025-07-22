from flax import linen
from brax.training.agents.ppo import networks_vision, networks
import mujoco
import numpy as np

import jax
from jax import numpy as jp
from brax.envs.base import Wrapper
from mujoco_playground._src import mjx_env

def custom_network_factory(obs_shape, action_size, preprocess_observations_fn,
                            activation_function='swish',
                            vision=False,
                            policy_hidden_layer_sizes=(256, 256),
                            value_hidden_layer_sizes=(256, 256),
                            ):
    if activation_function == 'swish':
        activation = linen.swish
    elif activation_function == 'relu':
        activation = linen.relu
    else:
        raise NotImplementedError(f'Not implemented anything for activation function {activation_function}')
    if vision:
        return networks_vision.make_ppo_networks_vision(
            observation_size=get_observation_size(),
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,  
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            normalise_channels=True            # Normalize image channels
        )
    else:
        return networks.make_ppo_networks(observation_size=get_observation_size(),
                                          action_size=action_size,
                                          preprocess_observations_fn=preprocess_observations_fn,
                                          policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                                          value_hidden_layer_sizes=value_hidden_layer_sizes,
                                          activation=activation)
    
def get_observation_size(vision_mode=None):
    if vision_mode is None:
        return 48
    elif vision_mode == 'rgb':
        return {
          "pixels/view_0": (120, 120, 3),  # RGB image
          "proprioception": (48,)          # Vector state
          }
    elif vision_mode == 'rgbd':
        return {
          "pixels/view_0": (120, 120, 4),  # RGBD image
          "proprioception": (48,)          # Vector state
      }
    elif vision_mode == 'rgb+depth':
        return {
          "pixels/view_0": (120, 120, 3),  # RGB image
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (48,)          # Vector state
          }
    else:
        raise NotImplementedError(f'No observation size known for "{vision_mode}"')
    

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


class AdaptiveTargetWrapper(Wrapper):
    """Automatically "resets" Brax envs that are done, without clearing curriculum state."""

    ## AutoReset Wrapper required to implement adaptive target curriculum; checks if episode is completed and calls reset inside this function;
    ## WARNING: Due to the following lines, applying the default Brax AutoResetWrapper has no effect to this env!

    #   def step(self, state: State, action: jax.Array) -> State:
    #     if 'steps' in state.info:
    #       steps = state.info['steps']
    #       steps = jp.where(state.done, jp.zeros_like(steps), steps)
    #       state.info.update(steps=steps)
    #     state = state.replace(done=jp.zeros_like(state.done))
    #     state = self.env.step(state, action)

    #     def where_done(x, y):
    #       done = state.done
    #       if done.shape:
    #         done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
    #       return jp.where(done, x, y)

    #     pipeline_state = jax.tree.map(
    #         where_done, state.info['first_pipeline_state'], state.pipeline_state
    #     )
    #     obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    #     return state.replace(pipeline_state=pipeline_state, obs=obs)

    def step(self, state: mjx_env.State, action: jax.Array):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        # ####################################################################################
        rng = state.info["rng"]

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        # obs_dict = self.get_obs_dict(state.pipeline_state, state.info)
        ## if isinstance(self.env, VmapWrapper) else functools.partial(self.env.reset_with_curriculum, rng=rng, info_before_reset=state.info)
        # if hasattr(self, 'batch_size'):  #if VmapWrapper is used (training mode)
        #     if self.batch_size is not None:
        #         rng = jax.random.split(rng, self.batch_size)
        #     state_after_reset = jax.vmap(self.env.reset_with_curriculum)(rng, state.info)
        # else:
        #     state_after_reset = self.env.reset_with_curriculum(rng, state.info)
        state_after_reset = jax.vmap(self.env.reset_with_curriculum)(rng, state.info)

        # fill state_after_reset.info with entries only in state.info (i.e. entries created by wrappers)
        for k in state.info:
            state_after_reset.info[k] = state_after_reset.info.get(k, state.info[k])

        data = jax.tree.map(
            where_done, state_after_reset.data, state.data  # state.pipeline_state
        )
        # obs_dict = jax.tree.map(where_done, obs_dict_after_reset, obs_dict)
        obs = jax.tree.map(where_done, state_after_reset.obs, state.obs)  # state.obs)
        info = jax.tree.map(where_done, state_after_reset.info, state.info)
        # ####################################################################################

        # ## Update state info of internal variables at the end of each env step
        # state.info['last_ctrl'] = obs_dict['last_ctrl']
        # state.info['motor_act'] = obs_dict['motor_act']
        # state.info['steps_since_last_hit'] = obs_dict['steps_since_last_hit']
        # state.info['steps_inside_target'] = obs_dict['steps_inside_target']
        # state.info['trial_idx'] = obs_dict['trial_idx'].copy()
        # if self.env.adaptive_task:
        #     state.info['trial_success_log_pointer_index'] = obs_dict['trial_success_log_pointer_index']
        #     state.info['trial_success_log'] = obs_dict['trial_success_log']
        # state.info['target_area_dynamic_width_scale'] = obs_dict['target_area_dynamic_width_scale']

        # # Also store variables useful for evaluation
        # state.info['target_success'] = obs_dict['target_success']
        # state.info['target_fail'] = obs_dict['target_fail']
        # state.info['task_completed'] = obs_dict['task_completed']

        # # finalize step
        # env_info = self.get_env_infos(state, data)

        # # update internal episode variables
        # # self._steps_inside_target = obs_dict['steps_inside_target']
        # # self._trial_idx += jp.select([obs_dict['target_success']], jp.ones(1))
        # # self._targets_hit += jp.select([obs_dict['target_success']], jp.ones(1))

        # # self._trial_success_log.at[self._trial_success_log_pointer_index] = jp.select([obs_dict['target_success'] | obs_dict['target_fail']], [(self._trial_success_log_pointer_index + 1) % self.success_log_buffer_length], self._trial_success_log_pointer_index)
        # # self._trial_success_log = jp.where(obs_dict['target_success'], jp.append(self._trial_success_log, 1), self._trial_success_log)  #TODO: ensure append adds entry to correct axis
        # # # # self._trial_idx += jp.select([_failure_condition], jp.ones(1))
        # # self._trial_success_log = jp.where(obs_dict['target_fail'], jp.append(self._trial_success_log, 0), self._trial_success_log)  #TODO: ensure append adds entry to correct axis

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=info,
        )
