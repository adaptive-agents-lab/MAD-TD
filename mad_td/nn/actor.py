from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from mad_td.nn.common import (
    MLP,
    EncoderDecoderResidualNetwork,
    DecoderResidualNetwork,
)
from mad_td.utils.jax import mish


class TanhActor(nn.Module):
    feature_dim: Sequence[int]
    output_dim: int
    activation_hidden: Callable[[jax.Array], jax.Array]
    activation_input: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states):
        x = MLP(
            output_dim=self.output_dim,
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=jnp.tanh,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(states)
        return x


class DecoderResidualActor(nn.Module):
    feature_dim: Sequence[int]
    output_dim: int
    num_blocks: int
    activation: Callable[[jax.Array], jax.Array] = mish

    @nn.compact
    def __call__(self, x):
        x = DecoderResidualNetwork(
            output_dim=self.output_dim,
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation,
            output_activation=jnp.tanh,
        )(x)
        return x


class EncoderDecoderResidualActor(nn.Module):
    feature_dim: Sequence[int]
    output_dim: int
    num_blocks: int
    activation: Callable[[jax.Array], jax.Array] = mish

    @nn.compact
    def __call__(self, x):
        x = EncoderDecoderResidualNetwork(
            output_dim=self.output_dim,
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation,
            output_activation=jnp.tanh,
        )(x)
        return x


class TDMPC2Actor(TanhActor):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Callable[[jax.Array], jax.Array] = mish
    normalize_input: bool = True
    normalize_hidden: bool = True
