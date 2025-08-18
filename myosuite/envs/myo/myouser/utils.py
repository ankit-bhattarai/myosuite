from typing import Callable, List, Optional, Sequence
import mujoco
from mujoco_playground import State
from mujoco_playground._src import mjx_env
import numpy as np

import mediapy as media


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
