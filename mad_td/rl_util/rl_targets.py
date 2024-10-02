from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from mad_td.nn.critic import resolve_distribution
from mad_td.utils.jax import two_hot, hl_gauss, symlog
from mad_td.cfgs.mad_td_config import MadTdHyperparams


def compute_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    critic: TrainState,
    actor: TrainState,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
):
    action = actor.apply_fn(actor.params, state)
    value = critic.apply_fn(critic.params, state, action)
    value = jnp.stack(value, axis=0)
    value = value.min(axis=0)
    td_target = reward + hyperparams.gamma * value
    return td_target


def compute_quantile_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    critic: TrainState,
    actor: TrainState,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
):
    action = actor.apply_fn(actor.params, state)
    value = critic.apply_fn(critic.params, state, action, get_distribution=True)
    value = jnp.stack(value, axis=-1)
    value = value.min(axis=-1)
    td_target = reward + hyperparams.gamma * value
    return td_target


def compute_c51_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    critic: TrainState,
    actor: TrainState,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
    discount: float | jax.Array | None = None,
):
    if discount is None:
        discount = hyperparams.gamma
    elif discount is not float and discount.shape == (1, 1):
        discount = discount[0]
    if reward.shape == (1, 1):
        reward = reward[0]

    action = actor.apply_fn(actor.params, state)
    value = critic.apply_fn(critic.params, state, action, get_distribution=True)
    value = jnp.stack(value, axis=-1)
    value = value.mean(axis=-1)  # TODO implement min?

    return histogram_target_shift(reward, value, discount, hyperparams)


def c51_target_from_states(
    state: jax.Array,
    reward: jax.Array,
    critic: TrainState,
    actor: TrainState,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
    discount: jax.Array = None,
):
    if discount is None:
        return jax.vmap(
            compute_c51_td_target_from_state,
            in_axes=[0, 0, None, None, None, None, None],
        )(state, reward, False, critic, actor, hyperparams, key)
    else:
        return jax.vmap(
            compute_c51_td_target_from_state,
            in_axes=[0, 0, None, None, None, None, 0],
        )(state, reward, critic, actor, hyperparams, key, discount)


def histogram_target_shift(
    reward: jax.Array,
    distribution: jax.Array,
    discount: jax.Array,
    hyperparams: MadTdHyperparams,
):
    # projection, inspiration source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_jax.py
    atoms = jnp.linspace(hyperparams.vmin, hyperparams.vmax, hyperparams.num_quantile)
    n_atoms = atoms.shape[0]

    # TODO(c51:log) next_atoms = symlog(reward[0] + hyperparams.gamma * symexp(atoms))
    next_atoms = reward + discount * atoms
    delta_z = atoms[1] - atoms[0]

    b = (next_atoms - hyperparams.vmin) / delta_z
    lower = jnp.clip(jnp.floor(b), a_min=0, a_max=n_atoms - 1)
    upper = jnp.clip(jnp.ceil(b), a_min=0, a_max=n_atoms - 1)
    # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
    d_m_l = (
        upper + (lower == upper).astype(jnp.float32) - b
    ) * distribution  # TODO(c51:log) log space for symexp
    d_m_u = (b - lower) * distribution
    td_target = jnp.zeros_like(distribution)

    def project_to_bins(i, val):
        val = val.at[lower[i].astype(jnp.int32)].add(d_m_l[i])
        val = val.at[upper[i].astype(jnp.int32)].add(d_m_u[i])
        return val

    td_target = jax.lax.fori_loop(0, td_target.shape[0], project_to_bins, td_target)
    return td_target


def bellman_residual_loss(
    state: jax.Array,
    action: jax.Array,
    value_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
    hyperparams: MadTdHyperparams,
) -> Tuple[jax.Array, Dict]:
    q1, q2 = critic.apply_fn(critic.params, state, action)
    critic_loss = jnp.mean(mask * (q1 - value_target) ** 2) + jnp.mean(
        mask * (q2 - value_target) ** 2
    )
    q1_log, q2_log = critic.apply_fn(critic.params, state, action)

    return (
        critic_loss,
        {  # type: ignore
            "critic_loss": critic_loss,
            "q1": q1_log.mean(),
            "q2": q2_log.mean(),
        },
    )


def crossentropy(pred, target, weight):
    return -jnp.mean(weight * (target * pred).mean(-1).reshape(weight.shape))


def binned_bellman_crossentropy_loss(
    state: jax.Array,
    action: jax.Array,
    value_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
    hyperparams: MadTdHyperparams,
) -> Tuple[jax.Array, Dict]:
    q1, q2 = critic.apply_fn(critic.params, state, action, get_logits=True)
    target_logits = value_target
    # target_logits is [batch, 1, num_critic_bins], num_critic_bins={num_bins in c51; 1 otherwise}
    if value_target.shape[-1] != q1.shape[-1]:  # target-logits are not distributional
        if hyperparams.use_symlog:
            value_target = symlog(value_target)
        if hyperparams.binned_regression_encoding.lower() == "hl-gauss":
            binned_regression_encoding_fun = hl_gauss
        elif hyperparams.binned_regression_encoding.lower() == "two-hot":
            binned_regression_encoding_fun = two_hot
        else:
            raise NotImplementedError(
                'binned_regression_encoding must be "hl-gauss" or "two-hot", but got '
                f"{hyperparams.binned_regression_encoding}"
            )
        target_logits = jax.vmap(
            binned_regression_encoding_fun, in_axes=[0, None, None, None]
        )(value_target, q1.shape[-1], hyperparams.vmin, hyperparams.vmax)
    # assert q1.shape == target_logits.shape, f"{q1.shape} != {target_logits.shape}"
    critic_loss = crossentropy(q1, target_logits, mask) + crossentropy(
        q2, target_logits, mask
    )

    q1_log = resolve_distribution(jax.nn.softmax(q1, axis=-1), hyperparams)
    q2_log = resolve_distribution(jax.nn.softmax(q2, axis=-1), hyperparams)

    l2_q_loss = jnp.mean(mask * jnp.square(q1_log - value_target))
    l2_q_loss += jnp.mean(mask * jnp.square(q2_log - value_target))

    return (
        critic_loss,
        {  # type: ignore
            "critic_loss": critic_loss,
            "q1": q1_log.mean(),
            "q1_std": q1_log.std(),
            "q2": q2_log.mean(),
            "q2_std": q2_log.std(),
            "l2_q_loss": l2_q_loss,
            "entropy": -(
                (jax.nn.softmax(q1, axis=-1) * q1).sum(axis=-1).mean()
                + (jax.nn.softmax(q2, axis=-1) * q2).sum(axis=-1).mean()
            )
            / 2,
        },
    )


def quantile_loss(pred, target, weight, num_quantiles):
    signed_loss = jnp.squeeze(target)[:, None, :] - jnp.squeeze(pred)[:, :, None]

    huber_loss = (
        # L2 loss when absolute error <= 1
        0.5 * jnp.int8((jnp.abs(signed_loss) <= 1)) * signed_loss**2
        +
        # L1 loss when absolute error > 1
        jnp.int8((jnp.abs(signed_loss) > 1)) * (jnp.abs(signed_loss) - 0.5)
    )

    tau = (jnp.arange(num_quantiles) + 0.5) / num_quantiles
    quantile_errors = jnp.abs(tau - jnp.int8(signed_loss < 0))
    quantile_huber_loss = huber_loss * quantile_errors

    # Take sum over quantiles and mean over next states (paper method)
    # loss = jnp.sum(jnp.mean(quantile_huber_loss, axis=2), axis=1)

    # Take mean over quantiles and next states (TODO: compare both?)
    loss = jnp.mean(jnp.mean(quantile_huber_loss, axis=2), axis=1)

    return jnp.mean(loss)


def quantile_bellman_residual_loss(
    state: jax.Array,
    action: jax.Array,
    quantile_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
    hyperparams: MadTdHyperparams,
) -> Tuple[jax.Array, Dict]:
    q1, q2 = critic.apply_fn(critic.params, state, action, get_distribution=True)

    num_quantile = q1.shape[2]

    critic_loss = quantile_loss(
        q1, quantile_target, mask, num_quantile
    ) + quantile_loss(q2, quantile_target, mask, num_quantile)

    return (
        critic_loss,
        {  # type: ignore
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
            "q1_quantiles_variance": q1.squeeze().var(-1).mean(),
            "q2_quantiles_variance": q2.squeeze().var(-1).mean(),
        },
    )


def soft_target_update(
    source: TrainState, target: TrainState, tau: float
) -> TrainState:
    source_params = source.params
    target_params = target.params
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: tau * x + (1 - tau) * y, target_params, source_params
    )

    new_target = TrainState.create(
        apply_fn=target.apply_fn,
        params=new_target_params,
        tx=target.tx,
    )
    return new_target
