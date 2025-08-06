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

"""Component modules for PPO networks."""

import jax.numpy as jnp
import flax.nnx as nnx
from typing import Optional, Callable, Sequence, Any


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        hidden_layers: Optional[Sequence[int]] = None,
        use_bias: bool = True,
        activation: Callable = nnx.leaky_relu,
    ):
        self.activation = activation
        if hidden_layers is None:
            self.layers = [
                nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)
            ]
            return
        assert type(hidden_layers) in [
            list,
            tuple,
        ], f"hidden_layers must be a list or tuple, got {type(hidden_layers)}"
        assert (
            len(hidden_layers) > 0
        ), f"hidden_layers must be a non-empty list, got {hidden_layers}"
        self.layers = [
            nnx.Linear(in_features, hidden_layers[0], use_bias=use_bias, rngs=rngs)
        ]
        for i in range(1, len(hidden_layers)):
            self.layers.append(
                nnx.Linear(
                    hidden_layers[i - 1], hidden_layers[i], use_bias=use_bias, rngs=rngs
                )
            )
        self.layers.append(
            nnx.Linear(hidden_layers[-1], out_features, use_bias=use_bias, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class StatesCombinerSimple(nnx.Module):
    """
    Simple states combiner that takes in a proprioception vector and outputs a normalised proprioception vector
    """

    def __init__(self, preprocess_observations_fn: Callable):
        self.preprocess_observations_fn = preprocess_observations_fn

    def __call__(self, x: jnp.ndarray, processor_params: Any):
        return self.preprocess_observations_fn(x, processor_params)


class StatesCombinerPredictStateVariables(nnx.Module):

    def __init__(self, preprocess_observations_fn: Callable):
        self.preprocess_observations_fn = preprocess_observations_fn

    def __call__(
        self,
        proprioception_feature: jnp.ndarray,
        vision_feature: jnp.ndarray,
        processor_params: Any,
    ):
        state_vector = jnp.concatenate(
            [proprioception_feature, vision_feature], axis=-1
        )
        state_vector = self.preprocess_observations_fn(state_vector, processor_params)
        return state_vector


class StatesCombinerVision(nnx.Module):
    def __init__(self, preprocess_observations_fn: Callable):
        self.preprocess_observations_fn = preprocess_observations_fn

    def __call__(
        self,
        proprioception_feature: jnp.ndarray,
        vision_feature: jnp.ndarray,
        processor_params: Any,
    ):
        proprioception_feature = self.preprocess_observations_fn(
            proprioception_feature, processor_params
        )
        state_vector = jnp.concatenate(
            [proprioception_feature, vision_feature], axis=-1
        )
        return state_vector
