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

"""PPO networks without vision."""

import flax.nnx as nnx
from brax.training import distribution
from typing import Callable, Sequence

from .base import PPONetworksUnifiedVision
from .components import MLP, StatesCombinerSimple


class NetworkNoVision(PPONetworksUnifiedVision):
    def __init__(
        self,
        proprioception_size: int,
        action_size: int,
        preprocess_observations_fn: Callable,
        rngs: nnx.Rngs,
        policy_hidden_layer_sizes: Sequence[int] = [32, 32, 32, 32],
        value_hidden_layer_sizes: Sequence[int] = [256, 256, 256, 256, 256],
    ):
        states_combiner = StatesCombinerSimple(preprocess_observations_fn)
        policy_network = MLP(
            proprioception_size,
            2 * action_size,
            hidden_layers=policy_hidden_layer_sizes,
            rngs=rngs,
        )
        value_network = MLP(
            proprioception_size, 1, hidden_layers=value_hidden_layer_sizes, rngs=rngs
        )
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
        proprioception_obs_key = "proprioception"
        super().__init__(
            states_combiner,
            policy_network,
            value_network,
            parametric_action_distribution,
            proprioception_obs_key,
        )


def make_ppo_networks_no_vision(
    proprioception_size: int,
    action_size: int,
    preprocess_observations_fn: Callable,
    policy_hidden_layer_sizes: Sequence[int] = [32, 32, 32, 32],
    value_hidden_layer_sizes: Sequence[int] = [256, 256, 256, 256, 256],
):
    model = nnx.bridge.to_linen(
        NetworkNoVision,
        proprioception_size=proprioception_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
    )
    return model
