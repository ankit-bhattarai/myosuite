
def get_observation_size(env_name, vision=False):
    if not vision:
      #TODO: infer full proprioception size from env (at least when vision is disabled)
      if env_name == "MyoUserPointing":
        return {'proprioception': 48}  #112}  #151}  #48}
      elif env_name == "MyoUserSteering":
        return {'proprioception': 63}
      else:
        raise NotImplementedError(f"No proprioception size available for env {env_name}")
    if vision == 'rgb':
      return {
          "pixels/view_0": (120, 120, 3),  # RGB image
          "proprioception": (44,)          # Vector state
          }
    elif vision == 'rgbd':
      return {
          "pixels/view_0": (120, 120, 4),  # RGBD image
          "proprioception": (44,)          # Vector state
      }
    elif vision == 'rgb+depth':
      return {
          "pixels/view_0": (120, 120, 3),  # RGB image
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (44,)          # Vector state
          }
    elif vision == 'rgbd_only':
      return {
          "pixels/view_0": (120, 120, 4),  # RGBD image
      }
    elif vision == 'depth_only':
      return {
          "pixels/depth": (120, 120, 1),  # Depth image
      }
    elif vision == 'depth':
      return {
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (44,)          # Vector state
      }
    elif vision == 'depth_w_aux_task':
      return {
          "pixels/depth": (120, 120, 1),  # Depth image
          "proprioception": (44,),          # Vector state
          "vision_aux_targets": (4,) # 3D target position + 1D target radius
      }
    else:
      raise NotImplementedError(f'No observation size known for "{vision}"')