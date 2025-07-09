from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable


@dataclass
class RendererArgs:
    gpu_id: int
    num_worlds: int
    render_width: int
    render_height: int
    enabled_geom_groups: List[int]
    enabled_cameras: List[int]
    use_rasterizer: bool
    
@dataclass
class AdaptiveTargetParams:
    init_target_area_width_scale: float
    adaptive_increase_success_rate: float
    adaptive_decrease_success_rate: float
    adaptive_change_step_size: float
    adaptive_change_min_trials: int
    success_log_buffer_length: int

@dataclass
class RewardWeights:
    reach: float
    bonus: float
    neural_effort: float

@dataclass
class TrainParams:
    n_train_steps: int
    n_eval_eps: int
    num_envs: int
    episode_length: int
    unroll_length: int
    num_minibatches: int
    num_updates_per_batch: int
    discounting: float
    learning_rate: float
    entropy_cost: float
    batch_size: int

@dataclass
class EnvParams:
    renderer_args: RendererArgs
    adaptive_target_params: AdaptiveTargetParams
    reward_weights: RewardWeights
    frame_skip: int
    seed: int
    model_path: str
    eval_mode: bool
    dwell_time_multiplier: float
    vision_enabled: bool
    vision_mode: str

@dataclass
class PointingTaskParams:
    target_pos_range_min: Tuple[float, float, float]
    target_pos_range_max: Tuple[float, float, float]
    target_radius_range: Tuple[float, float]
    ref_site: str
    reset_type: str



@dataclass
class NetworkParams:
    activation_fn: Callable
    policy_hidden_layer_sizes: List[int]
    value_hidden_layer_sizes: List[int]
    normalise_pixels: bool
    proprioception_obs_key: str
    proprioception_output_size: int
    vision_output_size: int

    
@dataclass
class RunParams:
    project_id: str
    experiment_id: str
    restore_params_path: str
    train_params: TrainParams
    network_params: NetworkParams
    pointing_task_params: PointingTaskParams
    env_params: EnvParams