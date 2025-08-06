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

"""Inference functions for PPO networks."""

from brax.training import distribution
from brax.training import types
from brax.training.types import PRNGKey
from typing import Tuple


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
