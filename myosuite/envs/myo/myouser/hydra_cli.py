from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any, List, Union, Dict, Callable

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from enum import Enum
from hydra.utils import instantiate, call
from myosuite.envs.myo.myouser.base import BaseEnvConfig, RLConfig
from myosuite.envs.myo.myouser.myouser_pointing_v0 import PointingEnvConfig
from myosuite.envs.myo.myouser.myouser_steering_v0 import SteeringEnvConfig

@dataclass
class WANDBEnabledConfig:
    enabled: bool = True
    entity: Union[str, None] = None # None by default so it uses the default entity override to choose a different entity
    suffix: Union[str, None] = None
    project: str = "MJXRL"
    tags: Union[List[str], None] = None
    group: Union[str, None] = None

@dataclass
class WANDBDisabledConfig:
    enabled: bool = False

class VisionModes(Enum):
    rgbd = 'rgbd'
    depth = 'depth'

@dataclass
class VisionEnabledConfig:
    enabled: bool = True
    vision_mode: VisionModes = field(default_factory=lambda: VisionModes.rgbd)
    gpu_id: int = 0
    num_worlds: int = '${rl.num_envs}'
    render_width: int = 120
    render_height: int = 120
    enabled_geom_groups: List[int] = field(default_factory=lambda: [0, 1, 2])
    enabled_cameras: List[int] = field(default_factory=lambda: [0])
    use_rasterizer: bool = False

@dataclass
class VisionDisabledConfig:
    enabled: bool = False

defaults = [
    {'wandb': 'enabled'},
    {'vision': 'disabled'},
    {'env': 'pointing'},
    {'rl': 'rl_config'},
    {'run': 'run'},
]

@dataclass
class RunConfig:
    seed: int = 0
    play_only: bool = False
    use_tb: bool = False
    rscope_envs: Union[int, None] = None
    deterministic_rscope: bool = True
    domain_randomization: bool = False

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    wandb: Any = MISSING
    vision: Any = MISSING
    env: BaseEnvConfig = MISSING
    rl: RLConfig = MISSING
    run: RunConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="wandb", name="enabled", node=WANDBEnabledConfig)
cs.store(group="wandb", name="disabled", node=WANDBDisabledConfig)
cs.store(group="vision", name="enabled", node=VisionEnabledConfig)
cs.store(group="vision", name="disabled", node=VisionDisabledConfig)
cs.store(group="env", name="base_env_config", node=BaseEnvConfig)
cs.store(group="env", name="pointing", node=PointingEnvConfig)
cs.store(group="env", name="steering", node=SteeringEnvConfig)
cs.store(group="rl", name="rl_config", node=RLConfig)
cs.store(group="run", name="run", node=RunConfig)

@hydra.main(version_base=None, config_name="config")
def my_app(cfg: Config) -> None:

    container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    print(OmegaConf.to_yaml(container))

if __name__ == "__main__":
    my_app()