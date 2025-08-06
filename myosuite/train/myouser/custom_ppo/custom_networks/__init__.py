# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks unified - re-exports for backward compatibility."""

# Import all components to maintain backward compatibility
from .base import PPONetworksUnifiedVision
from .components import (
    MLP,
    StatesCombinerSimple,
    StatesCombinerPredictStateVariables,
)
from .vision import (
    VisionEncoder,
    VisionAuxOutputIdentity,
    VisionAuxOutputStateVariables,
)
from .no_vision import (
    NetworkNoVision,
    make_ppo_networks_no_vision,
)
from .with_vision import (
    NetworkWithVision,
    make_ppo_networks_with_vision,
)
from .inference import make_inference_function_ppo_networks_unified

# Re-export everything for backward compatibility
__all__ = [
    # Base classes
    "PPONetworksUnifiedVision",
    # Component classes
    "MLP",
    "StatesCombinerSimple",
    "StatesCombinerPredictStateVariables",
    # Vision classes
    "VisionEncoder",
    "VisionAuxOutputIdentity",
    "VisionAuxOutputStateVariables",
    # Network implementations
    "NetworkNoVision",
    "NetworkWithVision",
    # Factory functions
    "make_ppo_networks_no_vision",
    "make_ppo_networks_with_vision",
    # Inference functions
    "make_inference_function_ppo_networks_unified",
]
