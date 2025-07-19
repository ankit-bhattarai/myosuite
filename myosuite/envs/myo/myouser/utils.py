from flax import linen
from brax.training.agents.ppo import networks_vision, networks
import mujoco
import numpy as np

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