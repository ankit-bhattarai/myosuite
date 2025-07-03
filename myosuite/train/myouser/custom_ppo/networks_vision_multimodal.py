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

"""PPO multimodal vision networks."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training.networks import FeedForwardNetwork, make_policy_network, make_value_network
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import functools
from dataclasses import dataclass
from brax.training.types import Params
import jax

ModuleDef = Any
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

@flax.struct.dataclass
class PPONetworksUnifiedExtractor:
  feature_extractor: FeedForwardNetwork
  policy_network: FeedForwardNetwork
  value_network: FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution

@flax.struct.dataclass
class ExtractorParams:
  cnn_encoder: Params
  cnn_decoder: Params
  state_encoder: Params
  inputs_fuser: Params

@flax.struct.dataclass
class UnifiedExtractor:
  cnn_normaliser: linen.Module
  cnn_encoder: FeedForwardNetwork
  cnn_decoder: FeedForwardNetwork
  state_encoder: FeedForwardNetwork
  inputs_fuser: FeedForwardNetwork
  init: Callable[..., Any]
  apply: Callable[..., Any]

@dataclass
class CNNLayer:
  features: int
  kernel_size: Tuple[int, int]
  stride: Tuple[int, int]
  padding: Tuple[int, int]
  use_bias: bool = True

class CNNNormaliser(linen.Module):
  normalise_pixels: bool = True
  @linen.compact
  def __call__(self, pixels_hidden: dict):
      if not self.normalise_pixels:
        return pixels_hidden
    # Calculates shared statistics over an entire 2D image.
      image_layernorm = functools.partial(
          linen.LayerNorm,
          use_bias=False,
          use_scale=False,
          reduction_axes=(-1, -2),
      )

      def ln_per_chan(v: jax.Array):
        normalised = [
            image_layernorm()(v[..., chan]) for chan in range(v.shape[-1])
        ]
        return jnp.stack(normalised, axis=-1)

      pixels_hidden = jax.tree.map(ln_per_chan, pixels_hidden)
      return pixels_hidden

class CNNDecoder(linen.Module):
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01)
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )

  @linen.compact
  def __call__(self, vision_features: jax.Array):
    hidden = vision_features

    hidden = linen.Dense(
      features=7200,
      use_bias=True,
    )(hidden)
    hidden = self.activation(hidden)

    # Unflatten
    spatial_dims = hidden.ndim - 1  # Number of leading dimensions to preserve

    # TODO: avoid hardcoding this and fix it properly
    n = 15
    hidden = jnp.reshape(hidden, hidden.shape[:spatial_dims] + (n, n, self.cnn_layers[0].features))
    # print(f'Shape of hidden after reshape layer: {hidden.shape}')

    for i in range(len(self.cnn_layers) - 1):
      current_layer = self.cnn_layers[i]
      next_layer_features = self.cnn_layers[i + 1].features
      hidden = linen.ConvTranspose(features=next_layer_features,
                                   kernel_size=current_layer.kernel_size,
                                   strides=current_layer.stride,
                                   use_bias=current_layer.use_bias,
                                   )(hidden)
      hidden = self.activation(hidden)
      # print(f'Shape of hidden after layer {i}: {hidden.shape}')

    # Last layer
    current_layer = self.cnn_layers[-1]
    hidden = linen.ConvTranspose(features=1,
                                 kernel_size=current_layer.kernel_size,
                                 strides=current_layer.stride,
                                 use_bias=current_layer.use_bias,
                                 )(hidden)
    # No activation for the last layer
    # print(f'Shape of hidden after last layer: {hidden.shape}')

    return hidden

def make_cnn_decoder(
  observation_size: Tuple[int, ...],
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  ),
) -> CNNDecoder:
  decoder = CNNDecoder(
    activation=activation,
    cnn_layers=cnn_layers,
  )
  def apply(decoder_params, vision_features):
    hidden = decoder.apply(decoder_params, vision_features)
    return hidden
  
  dummy_obs = jnp.zeros((1,) + observation_size)
  
  return FeedForwardNetwork(
    init=lambda key: decoder.init(key, dummy_obs), apply=apply
  )

class CNNEncoder(linen.Module):
  vision_output_size: int = 256
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01)
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )

  @linen.compact
  def __call__(self, pixels_hidden: dict):
    "Note that the input is normalised by the CNNNormaliser before being passed to the encoder."
    # Vision branch
    vision_features = []

    assert len(pixels_hidden) == 1, "Only one vision input is currently supported"

    for key in pixels_hidden:
    # CNN layers for vision
      hidden = pixels_hidden[key]

      for layer in self.cnn_layers:
        hidden = linen.Conv(
          features=layer.features,
          kernel_size=layer.kernel_size,
          strides=layer.stride,
          padding=layer.padding,
          use_bias=layer.use_bias,
        )(hidden)
        hidden = self.activation(hidden)
      
      # Flatten
      spatial_dims = hidden.ndim - 3  # Number of leading dimensions to preserve
      hidden = jnp.reshape(hidden, hidden.shape[:spatial_dims] + (-1,))
      vision_features.append(hidden)
      
    # Concatenate all vision features and pass through linear layer
    vision_concat = jnp.concatenate(vision_features, axis=-1)
    vision_out = linen.Dense(
        features=self.vision_output_size,
        use_bias=True,
    )(vision_concat)
    vision_out = self.activation(vision_out)

    return vision_out

def make_cnn_encoder(
  observation_size: Mapping[str, Tuple[int, ...]],
  vision_output_size: int = 256,
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )
) -> CNNEncoder:
  """Make CNN encoder."""
  cnn_encoder = CNNEncoder(
    vision_output_size=vision_output_size,
    activation=activation,
    cnn_layers=cnn_layers,
  )
  def apply(cnn_encoder_params, obs):
    features = cnn_encoder.apply(cnn_encoder_params, obs)
    return features
  
  dummy_obs = {
      key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
  }
  return FeedForwardNetwork(
      init=lambda key: cnn_encoder.init(key, dummy_obs), apply=apply
  )

class StateEncoder(linen.Module):
  """Encodes the proprioception state"""
  proprioception_output_size: int = 128
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01)

  @linen.compact
  def __call__(self, proprioception_input: jax.Array):
    proprioception_out = linen.Dense(
        features=self.proprioception_output_size,
        use_bias=True,
    )(proprioception_input)
    proprioception_out = self.activation(proprioception_out)
    return proprioception_out
  
def make_state_encoder(
  observation_size: Tuple[int, ...],
  proprioception_output_size: int = 128,
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
) -> StateEncoder:
  state_encoder = StateEncoder(
    proprioception_output_size=proprioception_output_size,
    activation=activation
  )
  def apply(processor_params,state_encoder_params, proprioception_input):
    proprioception_input = preprocess_observations_fn(proprioception_input, processor_params)
    proprioception_out = state_encoder.apply(state_encoder_params, proprioception_input)
    return proprioception_out
  
  dummy_obs = jnp.zeros((1,) + observation_size)
  return FeedForwardNetwork(
      init=lambda key: state_encoder.init(key, dummy_obs), apply=apply
  )

class InputsFuser(linen.Module):
  """Fuses vision and proprioception inputs."""
  combined_output_size: int = 384
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01)

  @linen.compact
  def __call__(self, vision_input, proprioception_input):
    vision_layernorm = linen.LayerNorm(use_bias=False, use_scale=False)
    proprioception_layernorm = linen.LayerNorm(use_bias=False, use_scale=False)

    vision_hidden = vision_layernorm(vision_input)
    proprioception_hidden = proprioception_layernorm(proprioception_input)

    hidden = jnp.concatenate([vision_hidden, proprioception_hidden], axis=-1)
    hidden = linen.Dense(
        features=self.combined_output_size,
        use_bias=True,
    )(hidden)
    hidden = self.activation(hidden)
    return hidden

def make_inputs_fuser(
    vision_observation_size: Tuple[int, ...],
    proprioception_observation_size: Tuple[int, ...],
    combined_output_size: int = 384,
    activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
) -> InputsFuser:
  inputs_fuser = InputsFuser(
    combined_output_size=combined_output_size,
    activation=activation,
  )

  def apply(inputs_fuser_params, vision_input, proprioception_input):
    hidden = inputs_fuser.apply(inputs_fuser_params, vision_input, proprioception_input)
    return hidden
  
  dummy_obs = (jnp.zeros((1,) + vision_observation_size), 
               jnp.zeros((1,) + proprioception_observation_size))
  return FeedForwardNetwork(
      init=lambda key: inputs_fuser.init(key, *dummy_obs), apply=apply
  )
 
class MultimodalFeatureExtractor(linen.Module):
  """Multimodal feature extractor with vision and proprioception branches."""
  
  vision_output_size: int = 256
  proprioception_output_size: int = 128
  fused_output_size: int = 384
  proprioception_obs_key: str = 'proprioception'
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01)
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )
  normalise_pixels: bool = True
  
  @linen.compact
  def __call__(self, data: dict):
    # Vision branch
    pixels_hidden = {k: v for k, v in data.items() if k.startswith('pixels/')}
  
    vision_out = CNNEncoder(activation=self.activation,
                     cnn_layers=self.cnn_layers,
                     normalise_pixels=self.normalise_pixels)(pixels_hidden)  
    
    # Proprioception branch
    assert self.proprioception_obs_key in data, "Proprioception input is required"
    
    proprioception_input = data[self.proprioception_obs_key]
    proprioception_out = linen.Dense(
        features=self.proprioception_output_size,
        use_bias=True,
    )(proprioception_input)
    proprioception_out = self.activation(proprioception_out)
    
    # Concatenate vision and proprioception features
    return jnp.concatenate([vision_out, proprioception_out], axis=-1)

def make_unified_feature_extractor(
    observation_size: Mapping[str, Tuple[int, ...]],
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    vision_output_size: int = 256,
    proprioception_output_size: int = 128,
    proprioception_obs_key: str = 'proprioception',
    fused_output_size: int = 384,
    activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
    cnn_layers: Sequence[CNNLayer] = (
      CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    ),
    normalise_pixels: bool = True,
) -> UnifiedExtractor:
  """Make unified extractor."""
  cnn_normaliser = CNNNormaliser(normalise_pixels=normalise_pixels)
  vision_observation_size = {k: v for k, v in observation_size.items() if k.startswith('pixels/')}
  assert len(vision_observation_size) == 1, "Only one vision observation is currently supported"
  assert proprioception_obs_key in observation_size, "Proprioception observation key must be in observation size"
  proprioception_observation_size = observation_size[proprioception_obs_key]
  cnn_encoder = make_cnn_encoder(observation_size=vision_observation_size,
                                 vision_output_size=vision_output_size,
                                 activation=activation,
                                 cnn_layers=cnn_layers)
  cnn_decoder = make_cnn_decoder(observation_size=(vision_output_size,),
                                 activation=activation,
                                 cnn_layers=cnn_layers[::-1]) # The encoder is reversed to get the decoder
  state_encoder = make_state_encoder(observation_size=proprioception_observation_size,
                                     proprioception_output_size=proprioception_output_size,
                                     activation=activation,
                                     preprocess_observations_fn=preprocess_observations_fn)
  inputs_fuser = make_inputs_fuser(vision_observation_size=(vision_output_size,),
                                   proprioception_observation_size=(proprioception_output_size,),
                                   combined_output_size=fused_output_size,
                                   activation=activation)
  
  def init(key):
    key_cnn_enc, key_cnn_dec, key_state_enc, key_inputs_fuser = jax.random.split(key, 4)
    cnn_encoder_params = cnn_encoder.init(key_cnn_enc)
    cnn_decoder_params = cnn_decoder.init(key_cnn_dec)
    state_encoder_params = state_encoder.init(key_state_enc)
    inputs_fuser_params = inputs_fuser.init(key_inputs_fuser)
    return ExtractorParams(
      cnn_encoder=cnn_encoder_params,
      cnn_decoder=cnn_decoder_params,
      state_encoder=state_encoder_params,
      inputs_fuser=inputs_fuser_params,
    )
  
  def apply(processor_params, params, obs):
    vision_observations = {k:v for k,v in obs.items() if k.startswith('pixels/')}
    assert len(vision_observations) == 1, "Only one vision observation is currently supported"
    proprioception_observations = obs[proprioception_obs_key]

    cnn_inputs = cnn_normaliser.apply({}, vision_observations)
    cnn_encoder_output = cnn_encoder.apply(params.cnn_encoder, cnn_inputs)
    # print(f"CNN encoder output shape: {cnn_encoder_output.shape}")
    cnn_decoder_output = cnn_decoder.apply(params.cnn_decoder, cnn_encoder_output)
    state_encoded = state_encoder.apply(processor_params, params.state_encoder, proprioception_observations)
    # print(f"State encoded shape: {state_encoded.shape}")
    fused_inputs = inputs_fuser.apply(params.inputs_fuser, cnn_encoder_output, state_encoded)
    # print(f"Fused inputs shape: {fused_inputs.shape}")
    image_input = cnn_inputs[list(cnn_inputs.keys())[0]]
    return {'combined_features': fused_inputs,
            'cnn_inputs': image_input,
            'cnn_decoder_preds': cnn_decoder_output,
            }
    
  return UnifiedExtractor(
    cnn_normaliser=cnn_normaliser,
    cnn_encoder=cnn_encoder,
    cnn_decoder=cnn_decoder,
    state_encoder=state_encoder,
    inputs_fuser=inputs_fuser,
    init=init,
    apply=apply,
  )

def make_ppo_networks_unified_extractor(
  observation_size: Mapping[str, Tuple[int, ...]],
  action_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  policy_hidden_layer_sizes: Sequence[int] = (256, 256),
  value_hidden_layer_sizes: Sequence[int] = (256, 256),
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
  normalise_pixels: bool = True,
  vision_output_size: int = 256,
  proprioception_output_size: int = 128,
  fused_output_size: int = 384,
  proprioception_obs_key: str = 'proprioception',
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )
) -> PPONetworksUnifiedExtractor:
  """Make PPO networks with unified feature extractor."""

  print(f"Making a PPO network with the following parameters:")
  print(f"Observation size: {observation_size}")
  print(f"Action size: {action_size}")
  print(f"Policy hidden layer sizes: {policy_hidden_layer_sizes}")
  print(f"Value hidden layer sizes: {value_hidden_layer_sizes}")
  print(f"Activation: {activation}")
  print(f"Normalise pixels: {normalise_pixels}")
  print(f"Vision output size: {vision_output_size}")
  print(f"Proprioception output size: {proprioception_output_size}")
  print(f"Fused output size: {fused_output_size}")
  print(f"Proprioception obs key: {proprioception_obs_key}")
  print(f"CNN layers: {cnn_layers}")
  print(f"Normalise pixels: {normalise_pixels}")

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )

  feature_extractor = make_unified_feature_extractor(
    observation_size=observation_size,
    preprocess_observations_fn=preprocess_observations_fn,
    fused_output_size=fused_output_size,
    vision_output_size=vision_output_size,
    proprioception_output_size=proprioception_output_size,
    proprioception_obs_key=proprioception_obs_key,
    activation=activation,
    cnn_layers=cnn_layers,
    normalise_pixels=normalise_pixels,
  )
  
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      fused_output_size,
      # Already preprocessed in the feature extractor so no need to preprocess again
      preprocess_observations_fn=types.identity_observation_preprocessor,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
  )
  value_network = make_value_network(
      fused_output_size,
      # Already preprocessed in the feature extractor so no need to preprocess again
      preprocess_observations_fn=types.identity_observation_preprocessor,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
  )

  return PPONetworksUnifiedExtractor(
    feature_extractor=feature_extractor,
    policy_network=policy_network,
    value_network=value_network,
    parametric_action_distribution=parametric_action_distribution,
  )


def make_inference_fn_unified_extractor(ppo_network: PPONetworksUnifiedExtractor):
  """Creates params and inference function for the PPO agent."""
  feature_extractor = ppo_network.feature_extractor
  policy_network = ppo_network.policy_network
  parametric_action_distribution = ppo_network.parametric_action_distribution
  
  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    processor_params = params[0]
    extractor_params = params[1]
    policy_params = params[2]
    
    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      extractor_outputs = feature_extractor.apply(processor_params, extractor_params, observations)
      # Policy network expects some kind of normalizer params, even though the network uses an identity preprocessor
      logits = policy_network.apply(None, policy_params, extractor_outputs['combined_features'])

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
          'log_prob': log_prob,
          'raw_action': raw_actions,
      }

    return policy

  return make_policy
    
    
  