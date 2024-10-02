import functools
from typing import Callable, Sequence, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from mad_td.utils.jax import torch_he_uniform, symexp, hl_gauss, mish


class Layer(nn.Module):
    size: int
    activation: Optional[Callable[[jax.Array], jax.Array]] = None
    add_norm: bool = False
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        x = nn.Dense(
            self.size,
            kernel_init=torch_he_uniform(size_param=self.init_weight),
        )(x)
        if self.add_norm:
            x = nn.LayerNorm()(x)
        if self.activation is not None:
            x = self.activation(x, **kwargs)
        return x


class MLP(nn.Module):
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
    def __call__(self, x: jax.Array, training: bool = False, **kwargs) -> jax.Array:
        inp_layer_size = self.feature_dim[0]
        hidden_layer_sizes = self.feature_dim[1:]
        output_layer_size = self.output_dim
        x = Layer(
            inp_layer_size,
            self.activation_input,
            add_norm=self.normalize_input,
            init_weight=self.init_weight,
        )(x)
        for size in hidden_layer_sizes:
            x = Layer(
                size,
                self.activation_hidden,
                add_norm=self.normalize_hidden,
                init_weight=self.init_weight,
            )(x)
        x = Layer(
            output_layer_size,
            self.activation_output,
            add_norm=self.normalize_output,
            init_weight=self.init_weight,
        )(x, **kwargs)
        return x


class ResidualBlock(nn.Module):
    feature_dim: int
    activation: Callable[[jax.Array], jax.Array] = mish

    @nn.compact
    def __call__(self, inp: jax.Array) -> jax.Array:
        x = Layer(self.feature_dim, self.activation, add_norm=True)(inp)
        x = Layer(self.feature_dim, add_norm=True)(x)
        return inp + x


class EncoderResidualNetwork(nn.Module):
    feature_dim: int
    num_blocks: int
    activation: Callable[[jax.Array], jax.Array] = mish

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = Layer(self.feature_dim, self.activation, add_norm=True)(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.feature_dim, self.activation)(x)
        return x


class DecoderResidualNetwork(nn.Module):
    feature_dim: int
    num_blocks: int
    output_dim: int
    activation: Callable[[jax.Array], jax.Array] = mish
    output_activation: Callable[[jax.Array], jax.Array] | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = Layer(self.feature_dim, self.activation, add_norm=True)(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.feature_dim, self.activation)(x)
        x = Layer(self.output_dim, self.output_activation)(x)
        return x


class EncoderDecoderResidualNetwork(nn.Module):
    feature_dim: int
    num_blocks: int
    output_dim: int
    activation: Callable[[jax.Array], jax.Array] = mish
    output_activation: Callable[[jax.Array], jax.Array] | None = None

    @nn.compact
    def __call__(self, x: jax.Array, get_logits=False) -> jax.Array:
        x = Layer(self.feature_dim, self.activation, add_norm=True)(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.feature_dim, self.activation)(x)
        x = Layer(self.output_dim, self.output_activation)(x, get_logits=get_logits)
        return x


class EnsembleMLP(nn.Module):
    feature_dim: Sequence[int]
    output_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
    num_ensemble: int

    @nn.compact
    def __call__(self, x):
        split_rngs = {"params": True}
        variable_axes = {"params": 0}

        VmapMLP = nn.vmap(
            functools.partial(
                MLP,
                output_dim=self.output_dim,
            ),
            split_rngs=split_rngs,
            variable_axes=variable_axes,
            in_axes=-2,
            out_axes=-2,
            axis_size=self.num_ensemble,
        )
        ys = VmapMLP(
            output_dim=self.output_dim,
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
        )(x)
        return ys


class BinnedNetwork(nn.Module):
    model: nn.Module
    min_value: float
    max_value: float
    add_zero_prior: bool
    use_symlog: bool

    @nn.compact
    def __call__(self, x, get_logits=False, get_distribution=False):
        # assert not (
        #     get_logits and get_distribution
        # ), "Cannot get both logits and distribution"
        x = self.model(x)
        num_bins = x.shape[-1]
        if self.add_zero_prior:
            zero_prior = hl_gauss(
                jnp.array([0]), num_bins, self.min_value, self.max_value
            )
            x = x + zero_prior * 17  # 17 is a magic number, seems to work well

        if get_logits:
            return jax.nn.log_softmax(x, axis=-1)

        elif get_distribution:
            return jax.nn.softmax(x, axis=-1)

        else:
            x = jax.nn.softmax(x, axis=-1)
            x = x.dot(jax.numpy.linspace(self.min_value, self.max_value, num_bins))
            if self.use_symlog:
                x = symexp(x)
            return x[..., jnp.newaxis]


class BinnedRegression(nn.Module):
    feature_dim: Sequence[int]
    num_bins: int
    min_value: float
    max_value: float
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
    add_zero_prior: bool
    use_symlog: bool
    use_residual: bool = False
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, x, get_logits=False, get_distribution=False):
        # assert not (
        #     get_logits and get_distribution
        # ), "Cannot get both logits and distribution"
        inp_layer_size = self.feature_dim[0]
        hidden_layer_sizes = self.feature_dim[1:]
        output_layer_size = self.num_bins
        x = Layer(
            inp_layer_size,
            self.activation_input,
            add_norm=self.normalize_input,
            init_weight=self.init_weight,
        )(x)
        for size in hidden_layer_sizes:
            x = Layer(
                size,
                self.activation_hidden,
                add_norm=self.normalize_hidden,
                init_weight=self.init_weight,
            )(x)
        x = Layer(
            output_layer_size,
            None,
            add_norm=self.normalize_output,
            init_weight=self.init_weight,
        )(x)

        if self.add_zero_prior:
            zero_prior = hl_gauss(
                jnp.array([0.0]), self.num_bins, self.min_value, self.max_value
            )
            x = x + zero_prior * 17  # 17 is a magic number, seems to work well

        if get_logits:
            return jax.nn.log_softmax(x, axis=-1)

        elif get_distribution:
            return jax.nn.softmax(x, axis=-1)

        else:
            x = jax.nn.softmax(x, axis=-1)
            x = x.dot(jax.numpy.linspace(self.min_value, self.max_value, self.num_bins))
            if self.use_symlog:
                x = symexp(x)
            return x[..., jnp.newaxis]
