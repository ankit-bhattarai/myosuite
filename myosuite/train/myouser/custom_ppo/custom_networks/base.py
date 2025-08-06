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

"""Base PPO networks unified."""

import jax.numpy as jnp
import jax
import flax.nnx as nnx
from brax.training import distribution  # TODO: get rid of brax dependency
from typing import Optional, Any


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
    stop_vision_gradient: bool

    def __init__(
        self,
        states_combiner: nnx.Module,
        policy_network: nnx.Module,
        value_network: nnx.Module,
        parametric_action_distribution: distribution.ParametricDistribution,
        proprioception_obs_key: str = "proprioception",
        vision_encoder: Optional[nnx.Module] = None,
        vision_aux_output: Optional[nnx.Module] = None,
        stop_vision_gradient: bool = False,
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
        self.stop_vision_gradient = stop_vision_gradient

    def get_values(self, state_vector):
        return jnp.squeeze(self.value_network(state_vector), axis=-1)

    def __call__(
        self,
        obs: dict,
        processor_params: Any,
        only_value_estimates: bool = False,
        only_policy_logits: bool = False,
        only_vision_aux_feature: bool = False,
    ):
        if self.has_vision:
            vision_feature = self.vision_encoder(obs)
            if only_vision_aux_feature:
                assert self.has_vision_aux_output, "Vision aux output must be provided if only_vision_aux_feature is True"
                return {"vision_aux_vector": self.vision_aux_output(vision_feature)}
            if self.stop_vision_gradient:
                print("Stopping vision gradient")
                pre_combined_vision_feature = jax.lax.stop_gradient(vision_feature)
            else:
                print("Not stopping vision gradient")
                pre_combined_vision_feature = vision_feature
            proprioception_feature = obs[self.proprioception_obs_key]
            state_vector = self.states_combiner(
                proprioception_feature,
                pre_combined_vision_feature,
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
