from flax import linen
from brax.training.agents.ppo import networks_vision, networks
import mujoco
import numpy as np

import jax
from jax import numpy as jp
from brax.envs.base import Wrapper
from mujoco_playground._src import mjx_env
    

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
