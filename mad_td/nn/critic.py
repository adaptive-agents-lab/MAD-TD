from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from mad_td.nn.common import (
    MLP,
    BinnedRegression,
    EncoderDecoderResidualNetwork,
    BinnedNetwork,
)
from mad_td.rl_types import AbstractCritic
from mad_td.cfgs.mad_td_config import MadTdHyperparams
from mad_td.utils.jax import mish, symexp


class Critic(AbstractCritic):
    feature_dim: Sequence[int]
    activation_input: Callable[[jax.Array], jax.Array]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)
        x = MLP(
            output_dim=1,
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=None,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(x)
        return x


class TwinnedCritic(AbstractCritic):
    feature_dim: Sequence[int]
    activation_input: Callable[[jax.Array], jax.Array]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    single_critic: bool = False
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, actions, get_logits=False, get_distribution=False):
        x1 = Critic(
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(states, actions)
        x2 = Critic(
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(states, actions)
        if self.single_critic:
            return x1, x1
        return x1, x2


def id(x):
    return x


class BinnedTDMPC2Critic(nn.Module):
    feature_dim: Sequence[int]
    num_bins: int
    vmin: float
    vmax: float
    add_zero_prior: bool = False
    use_symlog: bool = False
    single_critic: bool = False
    use_residual: bool = False
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, actions, get_logits=False, get_distribution=False):
        sa = jax.numpy.concatenate([states, actions], axis=-1)
        x1 = BinnedRegression(
            num_bins=self.num_bins,
            min_value=self.vmin,
            max_value=self.vmax,
            feature_dim=self.feature_dim,
            activation_hidden=mish,
            activation_input=mish,
            normalize_input=True,
            normalize_hidden=True,
            normalize_output=False,
            use_symlog=self.use_symlog,
            add_zero_prior=self.add_zero_prior,
            use_residual=self.use_residual,
            init_weight=self.init_weight,
        )(sa, get_logits=get_logits, get_distribution=get_distribution)
        x2 = BinnedRegression(
            num_bins=self.num_bins,
            min_value=self.vmin,
            max_value=self.vmax,
            feature_dim=self.feature_dim,
            activation_hidden=mish,
            activation_input=mish,
            normalize_input=True,
            normalize_hidden=True,
            normalize_output=False,
            use_symlog=self.use_symlog,
            add_zero_prior=self.add_zero_prior,
            use_residual=self.use_residual,
            init_weight=self.init_weight,
        )(sa, get_logits=get_logits, get_distribution=get_distribution)
        if self.single_critic:
            return x1, x1
        return x1, x2


class EncoderDecoderBinnedCritic(nn.Module):
    feature_dim: int
    num_bins: int
    vmin: float
    vmax: float
    num_blocks: int
    activation: Callable[[jax.Array], jax.Array] = mish
    add_zero_prior: bool = False
    use_symlog: bool = False
    single_critic: bool = False

    def setup(self):
        model1 = EncoderDecoderResidualNetwork(
            feature_dim=self.feature_dim,
            num_blocks=self.num_blocks,
            output_dim=self.num_bins,
            activation=self.activation,
            output_activation=None,
        )
        model2 = EncoderDecoderResidualNetwork(
            feature_dim=self.feature_dim,
            num_blocks=self.num_blocks,
            output_dim=self.num_bins,
            activation=self.activation,
            output_activation=None,
        )
        self.critic1 = BinnedNetwork(
            model1,
            min_value=self.vmin,
            max_value=self.vmax,
            add_zero_prior=self.add_zero_prior,
            use_symlog=self.use_symlog,
        )
        self.critic2 = BinnedNetwork(
            model2,
            min_value=self.vmin,
            max_value=self.vmax,
            add_zero_prior=self.add_zero_prior,
            use_symlog=self.use_symlog,
        )

    def __call__(self, states, actions, get_logits=False, get_distribution=False):
        sa = jax.numpy.concatenate([states, actions], axis=-1)
        x1 = self.critic1(sa, get_logits=get_logits, get_distribution=get_distribution)
        x2 = self.critic2(sa, get_logits=get_logits, get_distribution=get_distribution)
        if self.single_critic:
            return x1, x1
        return x1, x2


class TDMPCCritic(TwinnedCritic):
    activation_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    normalize_input: bool = True
    normalize_hidden: bool = False


class TDMPC2Critic(TwinnedCritic):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Callable[[jax.Array], jax.Array] = mish
    normalize_input: bool = True
    normalize_hidden: bool = True


def resolve_distribution(dist: jax.Array, hyperparams: MadTdHyperparams):
    if dist.shape[-1] == 1:
        return dist
    if hyperparams.quantile:
        return jnp.mean(dist, axis=-1, keepdim=True)
    else:
        x = dist.dot(
            jax.numpy.linspace(hyperparams.vmin, hyperparams.vmax, hyperparams.num_bins)
        )
        if hyperparams.use_symlog:
            x = symexp(x)
        return x[..., jnp.newaxis]
