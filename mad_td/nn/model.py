from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from mad_td.nn.common import (
    MLP,
    EncoderResidualNetwork,
    EncoderDecoderResidualNetwork,
)
from mad_td.utils.jax import mish, multi_softmax


class MLPEncoder(nn.Module):
    hidden_dim: Sequence[int]
    feature_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states):
        return MLP(
            feature_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(states)


class LatentModel(nn.Module):
    feature_dim: Sequence[int]
    output_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)

        true_reward = MLP(
            output_dim=1,
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=None,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )
        reward = true_reward(x)

        _forward = MLP(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )
        forward = _forward(x, get_logits=get_logits)

        return forward, reward


class IdentityEncoder(nn.Module):
    feature_dim: int

    def __call__(self, args, **kwargs):
        return args


class ResidualEncoder(nn.Module):
    feature_dim: int
    num_blocks: int
    activation: Callable[[jax.Array], jax.Array] = mish

    @nn.compact
    def __call__(self, states):
        x = EncoderResidualNetwork(
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation,
        )(states)
        return multi_softmax(x)


class ResidualLatentModel(nn.Module):
    feature_dim: int
    num_blocks: int
    output_dim: int
    activation: Callable[[jax.Array], jax.Array] = mish
    output_activation: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax

    def setup(self):
        self.forward_model = EncoderDecoderResidualNetwork(
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            output_dim=self.output_dim + 1,
            activation=self.activation,
            output_activation=None,
        )

    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)
        forward = self.forward_model(x, get_logits=get_logits)
        next_state = self.output_activation(forward[..., :-1], get_logits=get_logits)
        reward = forward[..., -1:]
        return next_state, reward


class TDMPCEncoder(MLPEncoder):
    activation_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class TDMPCLatentModel(LatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class TDMPC2Encoder(MLPEncoder):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True


class TDMPC2LatentModel(LatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True


class DummyLatentModel(LatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = lambda x: x
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True


class TDMPC2ContinuousEncoder(MLPEncoder):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = mish
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True


class TDMPC2ContinuousLatentModel(LatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True
