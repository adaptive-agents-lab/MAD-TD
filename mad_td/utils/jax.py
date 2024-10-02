from typing import Callable, Optional, Sequence, Dict, Union
import flax
from flax.training import train_state
import jax
import jax.numpy as jnp


PyTree = Union[jax.Array, Dict[str, "PyTree"]]


def batch_loss_fn(
    loss_fn: Callable,
    in_axes: Sequence[Union[int, None]] = (),
    out_axes: Sequence[Union[int, None]] = (),
    has_aux: bool = False,
) -> Callable:
    _batched_loss_fn = jax.vmap(loss_fn, in_axes=in_axes, out_axes=out_axes)

    def _f(*args):
        if has_aux:
            value, aux = _batched_loss_fn(*args)
            # return jnp.mean(value), jax.tree_map(lambda x: jnp.mean(x), aux)
            return jnp.mean(value), aux
        else:
            value = _batched_loss_fn(*args)
            return jnp.mean(value)

    return _f


# @jax.jit
def tree_list_mean(lot: Sequence[PyTree]) -> PyTree:
    return jax.tree_map(lambda *x: jnp.mean(jnp.stack(x, axis=0), axis=0), *lot)


def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
    size_param: float = 1.0,
):
    "TODO: push to jax"
    return jax.nn.initializers.variance_scaling(
        0.3333 * size_param,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def clamp_int(i, min_value, max_value):
    """Clamp an integer between min_value and max_value.

    Args:
        i (int): The integer to clamp.
        min_value (int): The minimum value.
        max_value (int): The maximum value.

    Returns:
        int: The clamped integer.
    """
    return int(max(min(i, max_value), min_value))


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class ExpandedTrainState(train_state.TrainState):
    variables: Optional[PyTree] = flax.core.frozen_dict.FrozenDict({})


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def multi_softmax(x, dim=8, get_logits=False):
    inp_shape = x.shape
    if dim is not None:
        x = x.reshape(-1, dim)
    if get_logits:
        x = jax.nn.log_softmax(x, axis=-1)
    else:
        x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*inp_shape)


def multi_log_softmax(x, dim=8):
    if dim is not None:
        return jax.nn.log_softmax(x.reshape(-1, dim), axis=-1).reshape(x.shape)
    else:
        return jax.nn.log_softmax(x, axis=-1)


def two_hot(inp, num_bins, vmin, vmax, epsilon=0.05):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = jnp.clip(inp, vmin, vmax).squeeze() / (1 - epsilon)
    bin_idx = jnp.floor((x - vmin) / bin_size)
    bin_offset = (x - vmin) / bin_size - bin_idx
    soft_two_hot = jnp.zeros(num_bins, dtype=jnp.float32)
    soft_two_hot = soft_two_hot.at[bin_idx.astype(jnp.int32)].set(1 - bin_offset)
    soft_two_hot = soft_two_hot.at[bin_idx.astype(jnp.int32) + 1].set(bin_offset)
    soft_two_hot = soft_two_hot.reshape(*inp.shape[:-1], num_bins)

    uniform = jnp.ones_like(soft_two_hot) / num_bins

    return (1 - epsilon) * soft_two_hot + epsilon * uniform


def hl_gauss(inp, num_bins, vmin, vmax, epsilon=0.0):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    x = jnp.clip(inp, vmin, vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(
        vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*inp.shape[:-1], num_bins)

    uniform = jnp.ones_like(target_probs) / num_bins

    return (1 - epsilon) * target_probs + epsilon * uniform


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)
