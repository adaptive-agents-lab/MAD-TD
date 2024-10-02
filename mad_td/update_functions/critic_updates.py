from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from mad_td.rl_types import RLBatch
from mad_td.rl_util.rl_targets import (
    bellman_residual_loss,
    binned_bellman_crossentropy_loss,
    quantile_bellman_residual_loss,
    compute_td_target_from_state,
    compute_quantile_td_target_from_state,
    compute_c51_td_target_from_state,
    soft_target_update,
)
from mad_td.cfgs.mad_td_config import (
    MadTdModels,
    MadTdHyperparams,
)
from mad_td.utils.adversarial import largest_eigenvalue


def get_target_fn(hyperparams: MadTdHyperparams):
    if hyperparams.c51:
        loss_f = binned_bellman_crossentropy_loss
        target_f = compute_c51_td_target_from_state
    elif hyperparams.quantile:
        loss_f = quantile_bellman_residual_loss
        target_f = compute_quantile_td_target_from_state
    else:
        # loss_f = binned_bellman_crossentropy_loss
        loss_f = (
            binned_bellman_crossentropy_loss
            if hyperparams.is_binned
            else bellman_residual_loss
        )
        target_f = compute_td_target_from_state
    return loss_f, target_f


def update_critic(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
) -> Tuple[MadTdModels, Dict]:
    loss_fn, target_fn = get_target_fn(hyperparams)

    state = batch.state
    action = batch.action[:, 0]
    reward = batch.reward[:, 0]
    next_state = batch.next_state[:, 0]
    mask = batch.mask[:, 0]

    loss_dict = {}

    # get BYOL aux reward
    if hyperparams.byol_explore:
        _latent_state = models.encoder.apply_fn(models.encoder.params, batch.state)
        _all_next_states = models.encoder.apply_fn(
            models.encoder.params, batch.next_state
        )
        _model_next_states = []
        for i in range(1):
            _latent_state, _ = models.latent_model.apply_fn(
                models.latent_model.params, _latent_state, batch.action[:, i]
            )
            _model_next_states.append(_latent_state)
        _model_next_states = jnp.stack(_model_next_states, axis=1)
        aux_loss = jnp.sum(
            jnp.mean(jnp.square(_model_next_states - _all_next_states), axis=-1),
            axis=-1,
        )[:, None]
        loss_dict["aux_reward"] = jnp.mean(aux_loss)
        loss_dict["aux_reward_std"] = jnp.std(aux_loss)
        loss_dict["aux_reward_ratio"] = jnp.mean(aux_loss) / jnp.mean(reward)
        reward = (
            reward
            + (hyperparams.byol_explore_weight / jnp.mean(aux_loss) * jnp.mean(reward))
            * aux_loss
        )

    # compute targets from real and from on policy model
    proportion_real = int(hyperparams.batch_size * hyperparams.proportion_real)

    real_latent_state = models.encoder.apply_fn(
        models.encoder.params, state[proportion_real:]
    )
    model_actions = models.actor.apply_fn(models.actor.params, real_latent_state)
    model_latent_next_state, model_reward = models.latent_model.apply_fn(
        models.latent_model.params, real_latent_state, model_actions
    )
    # model_latent_next_state = jnp.log(model_latent_next_state + 1e-8)

    real_latent_next_state = models.encoder_target.apply_fn(
        models.encoder_target.params, next_state[:proportion_real]
    )
    action = jnp.concatenate([action[:proportion_real], model_actions], axis=0)
    reward = jnp.concatenate([reward[:proportion_real], model_reward], axis=0)
    latent_next_state = jnp.concatenate(
        [real_latent_next_state, model_latent_next_state], axis=0
    )

    target_func = jax.vmap(target_fn, in_axes=(0, 0, None, None, None, 0))
    target_key, loss_key = jax.random.split(key, 2)
    key = jax.random.split(target_key, latent_next_state.shape[0])
    target = target_func(
        latent_next_state,
        reward,
        models.critic_target,
        models.actor,
        hyperparams,
        key,
    )

    latent_state = models.encoder.apply_fn(models.encoder.params, state)
    loss_func = jax.value_and_grad(
        # binned_bellman_crossentropy_loss,
        loss_fn,
        argnums=3,
        has_aux=True,
        allow_int=True,
    )

    (_, _loss_dict), grad = loss_func(
        latent_state,
        action,
        target,
        models.critic,
        mask,
        target_key,
        hyperparams,
    )
    loss_dict.update(_loss_dict)
    loss_dict["target_diversity"] = jnp.std(target)

    # logging OOD correlation
    full_latent = models.encoder.apply_fn(models.encoder.params, state)
    actor_actions = models.actor.apply_fn(models.actor.params, full_latent)
    action_dist = jnp.sqrt(
        jnp.sum(jnp.square(actor_actions - batch.action[:, 0]), axis=-1)
    )
    Q_values = models.critic.apply_fn(models.critic.params, full_latent, actor_actions)
    if hyperparams.average_critic_update:
        Q_values = jnp.mean(jnp.stack(Q_values, axis=1), axis=1)
    else:
        Q_values = jnp.min(jnp.stack(Q_values, axis=1), axis=1)

    loss_dict["ood_correlation"] = jnp.corrcoef(
        action_dist.squeeze(), Q_values.squeeze()
    )[0, 1]

    # update parameters
    new_critic = models.critic.apply_gradients(
        grads=grad.params,
    )
    new_models = MadTdModels(
        encoder=models.encoder,
        encoder_target=models.encoder_target,
        latent_model=models.latent_model,
        critic=new_critic,
        critic_target=soft_target_update(
            new_critic, models.critic_target, hyperparams.tau
        ),
        actor=models.actor,
    )

    # compute largest eigenvalues of the Hessian
    loss_dict["reward"] = jnp.mean(batch.reward[:, 0])
    loss_dict["reward_std"] = jnp.std(batch.reward[:, 0])

    return new_models, loss_dict
