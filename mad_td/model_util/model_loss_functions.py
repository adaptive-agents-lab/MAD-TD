import jax
import jax.numpy as jnp

from flax.training import train_state

from mad_td.cfgs.mad_td_config import MadTdHyperparams
from mad_td.update_functions.critic_updates import get_target_fn
from mad_td.utils.jax import multi_log_softmax, multi_softmax
from mad_td.nn.critic import resolve_distribution


def byol_crossent(
    pred: jax.Array,
    target: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    target = 0.0001 * (jnp.ones_like(target) / 8) + 0.9999 * target
    return -jnp.mean(weight * target * pred)


def cross_state_kl(
    pred: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    pred1 = pred[:, None]
    pred2 = pred[:, :, None]
    weight = jnp.concatenate([jnp.ones_like(weight[:, :1]), weight], axis=1)
    weight = weight[:, None] * weight[:, :, None]
    return jnp.mean(weight * (pred1 * jnp.log(pred2 + 1e-8)))


def crossent_daml(
    pred: jax.Array,
    target_value: jax.Array,
    target_action: jax.Array,
    critic: train_state.TrainState,
    weight: jax.Array,
) -> jax.Array:
    # pred_inp = jnp.log(pred + 1e-8)
    pred_inp = multi_softmax(pred)
    pred_values = critic.apply_fn(
        critic.params, pred_inp, target_action, get_logits=True
    )
    pred_values = jnp.stack(pred_values, 0)

    q_loss = -jnp.mean(weight * target_value * pred_values)

    return q_loss


def l2_vaml(
    pred: jax.Array,
    target_value: jax.Array,
    target_action: jax.Array,
    critic: train_state.TrainState,
    weight: jax.Array,
    hyperparams: MadTdHyperparams,
):
    pred_inp = multi_softmax(pred)
    pred_values = critic.apply_fn(
        critic.params, pred_inp, target_action, get_logits=False
    )
    pred_values = jnp.stack(pred_values, 0)

    target_value = resolve_distribution(target_value, hyperparams)
    pred_values = resolve_distribution(pred_values, hyperparams)

    return jnp.mean(weight * jnp.square(target_value - pred_values))


def l2_byol(
    pred: jax.Array,
    target: jax.Array,
    weight: jax.Array,
):
    pred = multi_softmax(pred)
    return jnp.mean(weight * jnp.square(target - pred))


def l2_logits(
    pred: jax.Array,
    target: jax.Array,
    weight: jax.Array,
):
    target = multi_log_softmax(target)
    return jnp.mean(weight * jnp.square(target - pred))


def l2_action(
    pred: jax.Array,
    target: jax.Array,
    actor: train_state.TrainState,
    weight: jax.Array,
):
    pred_inp = multi_softmax(pred)
    pred_action = actor.apply_fn(actor.params, pred_inp)
    return jnp.mean(weight * jnp.square(target - pred_action))


def compute_model_loss(
    state: jax.Array,
    actions: jax.Array,
    reward: jax.Array,
    next_states: jax.Array,
    mask: jax.Array,
    encoder: train_state.TrainState,
    encoder_target: train_state.TrainState,
    model: train_state.TrainState,
    critic: train_state.TrainState,
    critic_target: train_state.TrainState,
    actor: train_state.TrainState,
    keys: jax.Array,
    hyperparams: MadTdHyperparams,
):
    # fix all input shapes to one step
    # actions = actions[:, 0]
    # reward = reward[:, 0]
    # next_states = next_states[:, 0]
    # mask = mask[:, 0]

    # get state encoding
    first_latent_state: jax.Array = encoder.apply_fn(encoder.params, state)  # type: ignore
    latent_state = first_latent_state

    model_rewards_ = []
    model_next_states_ = []

    for i in range(hyperparams.length_training_rollout):
        keys, _ = jax.random.split(keys)
        latent_state, pred_rewards = model.apply_fn(
            model.params, latent_state, actions[:, i], get_logits=True
        )
        model_next_states_.append(latent_state)
        model_rewards_.append(pred_rewards)
        latent_state = multi_softmax(latent_state)

    model_next_states = jnp.stack(model_next_states_, axis=1)
    model_rewards = jnp.stack(model_rewards_, axis=1)

    # get real next states
    latent_next_states: jax.Array = jax.lax.stop_gradient(
        encoder_target.apply_fn(encoder_target.params, next_states)
    )  # type: ignore
    pred_action = actor.apply_fn(actor.params, latent_next_states)
    gt_value_dist = critic_target.apply_fn(
        critic_target.params, latent_next_states, pred_action, get_distribution=True
    )
    gt_value_dist = jnp.stack(gt_value_dist, axis=0)

    # get state_encoding for cross entropy maximization
    enc_next_latent_states = encoder.apply_fn(encoder.params, next_states)
    all_latent_states = jnp.concatenate(
        [first_latent_state[:, None], enc_next_latent_states], axis=1
    )
    cross_state_loss = cross_state_kl(all_latent_states, mask)

    # temporary_logging
    value = critic.apply_fn(critic.params, latent_next_states, pred_action)
    value = jnp.stack(value, axis=0)

    # compute losses
    reward_loss = jnp.mean(mask * (reward - model_rewards) ** 2)
    total_loss = reward_loss + cross_state_loss * hyperparams.cross_kl_weight
    model_loss = jax.vmap(byol_crossent, in_axes=(1, 1, 1))(
        model_next_states,
        latent_next_states,
        mask,
    ).mean()
    total_loss += model_loss

    if hyperparams.use_daml:
        vaml_loss = jax.vmap(crossent_daml, in_axes=(1, 2, 1, None, 1))(
            model_next_states,
            gt_value_dist,
            pred_action,
            critic_target,
            mask,
        ).mean()
        total_loss += vaml_loss
    else:
        vaml_loss = jnp.zeros_like(model_loss)

    if hyperparams.update_encoder_with_critic_loss:
        # get BYOL aux reward
        if hyperparams.byol_explore:
            aux_loss = jnp.mean(
                jnp.square(enc_next_latent_states[:, 0] - model_next_states[:, 0]),
                axis=-1,
            )[:, None]
            reward = (
                reward[:, 0]
                + (
                    hyperparams.byol_explore_weight
                    / jnp.mean(aux_loss)
                    * jnp.mean(reward[:, 0])
                )
                * aux_loss
            )
        else:
            reward = reward[:, 0]
        loss_fn, target_fn = get_target_fn(hyperparams)
        target_key, _ = jax.random.split(keys)
        target_key = jax.random.split(target_key, latent_next_states.shape[0])

        target_fn = jax.vmap(target_fn, in_axes=(0, 0, None, None, None, 0))
        target = jax.lax.stop_gradient(
            target_fn(
                latent_next_states[:, 0],
                reward,
                critic_target,
                actor,
                hyperparams,
                target_key,
            )
        )
        critic_loss, _ = loss_fn(
            first_latent_state,
            actions[:, 0],
            target,
            critic,
            mask[:, 0],
            keys,
            hyperparams,
        )
        total_loss += critic_loss
    else:
        critic_loss = jnp.zeros_like(total_loss)

    if not hyperparams.use_model_loss:
        total_loss = jnp.zeros_like(total_loss)
        model_loss = jnp.zeros_like(model_loss)
        reward_loss = jnp.zeros_like(reward_loss)
        vaml_loss = jnp.zeros_like(vaml_loss)
        critic_loss = jnp.zeros_like(critic_loss)

    losses = {
        "reward_loss": reward_loss,
        "model_loss": model_loss,
        "vaml_loss": vaml_loss,
        "model_critic_loss": critic_loss,
        "q_mean": jnp.mean(value),
        "encoder_entropy": -jnp.mean(
            first_latent_state * jnp.log(first_latent_state + 1e-8)
        )
        * 8,  # factor 8 due to the multi_softmax
        "l2_byol": jax.vmap(l2_byol, in_axes=(1, 1, 1))(
            model_next_states, latent_next_states, mask
        ),
        "l2_logits": jax.vmap(l2_logits, in_axes=(1, 1, 1))(
            model_next_states, latent_next_states, mask
        ),
        "l2_action": jax.vmap(l2_action, in_axes=(1, 1, None, 1))(
            model_next_states, pred_action, actor, mask
        ),
        "l2_vaml": jax.vmap(l2_vaml, in_axes=(1, 2, 1, None, 1, None))(
            model_next_states,
            gt_value_dist,
            pred_action,
            critic_target,
            mask,
            hyperparams,
        ),
    }

    return total_loss, {"total_loss": total_loss, **losses}
