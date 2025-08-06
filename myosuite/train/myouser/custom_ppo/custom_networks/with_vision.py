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

"""PPO networks with vision."""

import flax.nnx as nnx
from brax.training import distribution
from typing import Callable, Sequence, Optional

from .base import PPONetworksUnifiedVision
from .components import MLP, StatesCombinerPredictStateVariables, StatesCombinerVision
from .vision import VisionEncoder, VisionAuxOutputIdentity


class NetworkWithVision(PPONetworksUnifiedVision):
    def __init__(
        self,
        proprioception_size: int,
        action_size: int,
        encoder_out_size: int,
        preprocess_observations_fn: Callable,
        rngs: nnx.Rngs,
        cheat_vision_aux_output: bool = False,
        policy_hidden_layer_sizes: Sequence[int] = [32, 32, 32, 32],
        value_hidden_layer_sizes: Sequence[int] = [256, 256, 256, 256, 256],
        has_vision_aux_output: bool = False,
        vision_aux_output_mlp: bool = False,
        vision_aux_output_mlp_output_size: Optional[int] = None,
        vision_encoder_normalize_output: bool = True,
        stop_vision_gradient: bool = False,
    ):
        states_combiner = StatesCombinerVision(
            preprocess_observations_fn=preprocess_observations_fn
        )
        state_vector_size = proprioception_size + encoder_out_size
        policy_network = MLP(
            state_vector_size,
            2 * action_size,
            hidden_layers=policy_hidden_layer_sizes,
            rngs=rngs,
        )
        value_network = MLP(
            state_vector_size, 1, hidden_layers=value_hidden_layer_sizes, rngs=rngs
        )
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
        proprioception_obs_key = "proprioception"
        vision_encoder = VisionEncoder(
            rngs=rngs, cheat_vision_aux_output=cheat_vision_aux_output,
            mlp_out_size=encoder_out_size,
            normalize_output=vision_encoder_normalize_output
        )
        if has_vision_aux_output:
            if vision_aux_output_mlp:
                assert vision_aux_output_mlp_output_size is not None, "vision_aux_output_mlp_output_size must be provided if vision_aux_output_mlp is True"
                vision_aux_output = MLP(
                    encoder_out_size,
                    vision_aux_output_mlp_output_size,
                    rngs=rngs
                )
            else:
                vision_aux_output = VisionAuxOutputIdentity(rngs=rngs)
        else:
            vision_aux_output = None
        super().__init__(
            states_combiner,
            policy_network,
            value_network,
            parametric_action_distribution,
            proprioception_obs_key,
            vision_encoder,
            vision_aux_output,
            stop_vision_gradient=stop_vision_gradient,
        )


def make_ppo_networks_with_vision(
    proprioception_size: int,
    action_size: int,
    encoder_out_size: int,
    preprocess_observations_fn: Callable,
    cheat_vision_aux_output: bool = False,
    has_vision_aux_output: bool = False,
    vision_aux_output_mlp: bool = False,
    vision_aux_output_mlp_output_size: Optional[int] = None,
    vision_encoder_normalize_output: bool = True,
    stop_vision_gradient: bool = False,
):
    model = nnx.bridge.to_linen(
        NetworkWithVision,
        proprioception_size=proprioception_size,
        action_size=action_size,
        encoder_out_size=encoder_out_size,
        preprocess_observations_fn=preprocess_observations_fn,
        cheat_vision_aux_output=cheat_vision_aux_output,
        has_vision_aux_output=has_vision_aux_output,
        vision_aux_output_mlp=vision_aux_output_mlp,
        vision_aux_output_mlp_output_size=vision_aux_output_mlp_output_size,
        vision_encoder_normalize_output=vision_encoder_normalize_output,
        stop_vision_gradient=stop_vision_gradient,
    )
    return model
