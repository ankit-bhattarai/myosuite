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

"""Vision-related modules for PPO networks."""

import jax.numpy as jnp
import flax.nnx as nnx
from typing import Optional, Callable, Sequence


class VisionEncoder(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        c_hid: int = 32,
        vision_out_size: int = 14400,
        mlp_between_size: int = 256,
        mlp_out_size: int = 4,
        mlp_hidden_layers: Optional[Sequence[int]] = [256, 128, 64, 32],
        cheat_vision_aux_output: bool = False,
        use_bias: bool = True,
        activation: Callable = nnx.leaky_relu,
    ):
        if cheat_vision_aux_output:
            self.cheat_vision_aux_output = True
            return
        else:
            self.cheat_vision_aux_output = False
        self.conv1 = nnx.Conv(
            1, c_hid, kernel_size=(3, 3), strides=2, rngs=rngs, use_bias=use_bias
        )
        self.conv2 = nnx.Conv(
            c_hid, c_hid, kernel_size=(3, 3), rngs=rngs, use_bias=use_bias
        )
        self.conv3 = nnx.Conv(
            c_hid,
            2 * c_hid,
            kernel_size=(3, 3),
            strides=2,
            rngs=rngs,
            use_bias=use_bias,
        )
        self.conv4 = nnx.Conv(
            2 * c_hid, 2 * c_hid, kernel_size=(3, 3), rngs=rngs, use_bias=use_bias
        )
        self.conv5 = nnx.Conv(
            2 * c_hid,
            2 * c_hid,
            kernel_size=(3, 3),
            strides=2,
            rngs=rngs,
            use_bias=use_bias,
        )
        self.linear1 = nnx.Linear(vision_out_size, mlp_between_size, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_between_size, mlp_out_size, rngs=rngs)
        # self.mlp = MLP(
        #     vision_out_size,
        #     mlp_out_size,
        #     rngs=rngs,
        #     hidden_layers=mlp_hidden_layers,
        #     use_bias=use_bias,
        #     activation=activation,
        # )
        self.activation = activation

    def __call__(self, x: dict):
        if self.cheat_vision_aux_output:
            return x["vision_aux_targets"]
        vision_keys = [k for k in x.keys() if k.startswith("pixels/")]
        assert len(vision_keys) == 1, "Only one vision key is supported"
        x = x[vision_keys[0]]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        spatial_dims = x.ndim - 3
        x = x.reshape(*x.shape[:spatial_dims], -1)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        # x = self.mlp(x)
        return x


class VisionAuxOutputIdentity(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        pass

    def __call__(self, x: jnp.ndarray):
        return x


class VisionAuxOutputStateVariables(nnx.Module):
    def __init__(self, num_state_variables: int, rngs: nnx.Rngs):
        self.num_state_variables = num_state_variables

    def __call__(self, x: jnp.ndarray):
        """Input will be something like [batch_size, unroll_length, vision_out] or [num_envs, vision_out]
        I only want to extract a vector of shape [batch_size, unroll_length, num_state_variables] or [num_envs, num_state_variables]
        """
        return x[..., : self.num_state_variables]
