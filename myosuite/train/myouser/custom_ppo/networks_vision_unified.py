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

"""PPO networks unified."""

import jax.numpy as jnp
import jax
import flax.nnx as nnx
from brax.training import distribution  # TODO: get rid of brax dependency
from brax.training import types
from brax.training.types import PRNGKey

from typing import Optional, Callable, Sequence, Any, Tuple


class PPONetworksUnifiedVision(nnx.Module):
    """
    There are three possible scenarios in which the network can be used

    1. No vision present:
     - The input which is a simple vector of proprioception is processed by the states_combiner and then directly
        passed to the policy and value networks.

    2. Vision present without supplementary task:
     - The vision encoder encodes the vision input into a lower dimensional feature vector.
     - The feature vector is fed into the combiner which takes that along with the proprioception vector and outputs a
        unified state vector
     - The unified state encoder is used as input to the policy and value nets respectively

    3. Vision present with supplementary task
    - Supplementary tasks can include predicting state variables or reconstructing the original input
    - The vision encoder's feature vectors are passed into the combiner in the same way as in (2).
    - In addition the vision_aux_output module takes in the output of the vision encoder and predicts variable(s)
        which allow the vision encoder
        to be trained using supervised learning.

    In all cases, the network will be used as follows:
    I. get_unified_state_vector(obs) ->
        This gives vectors which can be directly passed on to the policy and value networks as well as possibly used
        for aux training tasks
        Case 1: {'state': proprioception_vector}
        Case 2: {'state': unified_state_vector}
        Case 3: {'state': unified_state_vector, 'supp_vector': output_from_vision_aux_output}
    II. policy_network.apply(unified_state_vector) & value_network.apply(unified_state_vector)
        This gives the network's outputs for policy and value

    The schemas for the networks are as follows:
    states_combiner:
        Case 1: Input: proprioception_vector, Output: normalised proprioception_vector
        Case 2/3: Input: proprioception_vector, vision_feature, Output: state_vector
    policy_network:
        Input: state_vector, Output: policy_logits
    value_network:
        Input: state_vector, Output: value_estimates
    parametric_action_distribution: not used by this network at all
    vision_encoder:
        Input: vision_obs, Output: vision_feature
    vision_aux_output:
        Input: vision_feature, Output: vision_aux_vector
    """

    states_combiner: nnx.Module
    policy_network: nnx.Module
    value_network: nnx.Module
    parametric_action_distribution: distribution.ParametricDistribution
    proprioception_obs_key: str
    vision_encoder: Optional[nnx.Module]
    vision_aux_output: Optional[nnx.Module]
    has_vision: bool
    has_vision_aux_output: bool

    def __init__(
        self,
        states_combiner: nnx.Module,
        policy_network: nnx.Module,
        value_network: nnx.Module,
        parametric_action_distribution: distribution.ParametricDistribution,
        proprioception_obs_key: str = "proprioception",
        vision_encoder: Optional[nnx.Module] = None,
        vision_aux_output: Optional[nnx.Module] = None,
    ):
        self.states_combiner = states_combiner
        self.policy_network = policy_network
        self.value_network = value_network
        self.parametric_action_distribution = parametric_action_distribution
        self.has_vision = vision_encoder is not None
        self.has_vision_aux_output = vision_aux_output is not None
        if self.has_vision_aux_output:
            assert (
                self.has_vision
            ), "Vision encoder must be provided if vision_aux_output is provided"
        self.vision_encoder = vision_encoder
        self.vision_aux_output = vision_aux_output
        self.states_combiner = states_combiner
        self.proprioception_obs_key = proprioception_obs_key

    def get_values(self, state_vector):
        return jnp.squeeze(self.value_network(state_vector), axis=-1)

    def __call__(
        self,
        obs: dict,
        processor_params: Any,
        only_value_estimates: bool = False,
        only_policy_logits: bool = False,
    ):
        if self.has_vision:
            vision_obs = {k: v for k, v in obs.items() if k.startswith("pixels/")}
            vision_feature = self.vision_encoder(vision_obs)
            gradient_stopped_vision_feature = jax.lax.stop_gradient(vision_feature)
            proprioception_feature = obs[self.proprioception_obs_key]
            state_vector = self.states_combiner(
                proprioception_feature,
                gradient_stopped_vision_feature,
                processor_params=processor_params,
            )
        else:
            state_vector = self.states_combiner(
                obs[self.proprioception_obs_key], processor_params=processor_params
            )

        if only_policy_logits and only_value_estimates:
            raise ValueError(
                "only_policy_logits and only_value_estimates cannot be True at the same time"
            )

        if only_policy_logits:
            return {"policy_logits": self.policy_network(state_vector)}
        if only_value_estimates:
            return {"value_estimates": self.get_values(state_vector)}

        value_estimates = self.get_values(state_vector)
        policy_logits = self.policy_network(state_vector)
        if not self.has_vision_aux_output:
            return {"policy_logits": policy_logits, "value_estimates": value_estimates}

        vision_aux_vector = self.vision_aux_output(vision_feature)
        return {
            "policy_logits": policy_logits,
            "value_estimates": value_estimates,
            "vision_aux_vector": vision_aux_vector,
        }


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
        assert (
            type(hidden_layers) == list
        ), f"hidden_layers must be a list, got {type(hidden_layers)}"
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


class NetworkNoVision(PPONetworksUnifiedVision):
    def __init__(
        self,
        proprioception_size: int,
        action_size: int,
        preprocess_observations_fn: Callable,
        rngs: nnx.Rngs,
    ):
        states_combiner = StatesCombinerSimple(preprocess_observations_fn)
        policy_network = MLP(
            proprioception_size,
            2 * action_size,
            hidden_layers=[32, 32, 32, 32],
            rngs=rngs,
        )
        value_network = MLP(
            proprioception_size, 1, hidden_layers=[256, 256, 256, 256, 256], rngs=rngs
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
    proprioception_size: int, action_size: int, preprocess_observations_fn: Callable
):
    model = nnx.bridge.to_linen(
        NetworkNoVision,
        proprioception_size=proprioception_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    return model


def make_inference_function_ppo_networks_unified(network):
    network = network
    assert (
        "action_size" in network.kwargs
    ), "action_size must be provided to make_inference_function_simple_unified_network"
    action_size = network.kwargs["action_size"]
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        processor_params, model_params = params

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = network.apply(
                model_params,
                observations,
                processor_params=processor_params,
                only_policy_logits=True,
            )["policy_logits"]
            if deterministic:
                return parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


class VisionEncoder(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        vision_out_size: int = 7200,
        mlp_out_size: int = 4,
        mlp_hidden_layers: Optional[Sequence[int]] = [256],
        use_bias: bool = True,
        activation: Callable = nnx.leaky_relu,
    ):
        self.conv1 = nnx.Conv(
            1, 8, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), rngs=rngs
        )
        self.conv2 = nnx.Conv(
            8, 16, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), rngs=rngs
        )
        self.conv3 = nnx.Conv(
            16, 32, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), rngs=rngs
        )
        self.mlp = MLP(
            vision_out_size,
            mlp_out_size,
            rngs=rngs,
            hidden_layers=mlp_hidden_layers,
            use_bias=use_bias,
            activation=activation,
        )
        self.activation = activation

    def __call__(self, x: jnp.ndarray):
        vision_keys = [k for k in x.keys() if k.startswith("pixels/")]
        assert len(vision_keys) == 1, "Only one vision key is supported"
        x = x[vision_keys[0]]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        spatial_dims = x.ndim - 3
        x = x.reshape(*x.shape[:spatial_dims], -1)
        x = self.mlp(x)
        return x


class VisionAuxOutputIdentity(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        pass

    def __call__(self, x: jnp.ndarray):
        return x


class StatesCombinerPredictStateVariables(nnx.Module):

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


class NetworkWithVision(PPONetworksUnifiedVision):
    def __init__(
        self,
        proprioception_size: int,
        action_size: int,
        encoder_out_size: int,
        preprocess_observations_fn: Callable,
        rngs: nnx.Rngs,
    ):
        states_combiner = StatesCombinerPredictStateVariables(
            preprocess_observations_fn
        )
        state_vector_size = proprioception_size + encoder_out_size
        policy_network = MLP(
            state_vector_size,
            2 * action_size,
            hidden_layers=[32, 32, 32, 32],
            rngs=rngs,
        )
        value_network = MLP(
            state_vector_size, 1, hidden_layers=[256, 256, 256, 256, 256], rngs=rngs
        )
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
        proprioception_obs_key = "proprioception"
        vision_encoder = VisionEncoder(rngs=rngs)
        vision_aux_output = VisionAuxOutputIdentity(rngs=rngs)
        states_combiner = StatesCombinerPredictStateVariables(
            preprocess_observations_fn
        )
        super().__init__(
            states_combiner,
            policy_network,
            value_network,
            parametric_action_distribution,
            proprioception_obs_key,
            vision_encoder,
            vision_aux_output,
        )


def make_ppo_networks_with_vision(
    proprioception_size: int,
    action_size: int,
    encoder_out_size: int,
    preprocess_observations_fn: Callable,
):
    model = nnx.bridge.to_linen(
        NetworkWithVision,
        proprioception_size=proprioception_size,
        action_size=action_size,
        encoder_out_size=encoder_out_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    return model
