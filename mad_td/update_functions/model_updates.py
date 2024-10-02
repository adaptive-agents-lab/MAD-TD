import jax
from mad_td.rl_util.rl_targets import soft_target_update

from mad_td.rl_types import RLBatch
from mad_td.model_util.model_loss_functions import compute_model_loss
from mad_td.cfgs.mad_td_config import (
    MadTdHyperparams,
    MadTdModels,
)


def update_model(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
):
    state = batch.state
    action = batch.action
    reward = batch.reward
    next_state = batch.next_state
    mask = batch.mask

    batched_loss_function = compute_model_loss
    loss_grad = jax.value_and_grad(
        batched_loss_function,
        argnums=(5, 7),
        has_aux=True,
        allow_int=True,
    )
    ((_, aux), grads) = loss_grad(
        state,
        action,
        reward,
        next_state,
        mask,
        models.encoder,
        models.encoder_target,
        models.latent_model,
        models.critic,
        models.critic_target,
        models.actor,
        key,
        hyperparams,
    )
    loss_dict = aux

    # log encoder statistics
    latent = models.encoder.apply_fn(models.encoder.params, state)
    latent_std = latent.std(axis=0).mean()
    latent_mean = latent.mean()
    encoder_statistics = {
        "latent_std": latent_std,
        "latent_mean": latent_mean,
    }

    # update parameters
    new_encoder = models.encoder.apply_gradients(grads=grads[0].params)
    new_latent_model = models.latent_model.apply_gradients(grads=grads[1].params)
    new_models = MadTdModels(
        encoder=new_encoder,
        encoder_target=soft_target_update(
            new_encoder,
            models.encoder_target,
            hyperparams.tau if hyperparams.use_target_encoder else 0.0,
        ),
        latent_model=new_latent_model,
        critic=models.critic,
        actor=models.actor,
        critic_target=models.critic_target,
    )
    return new_models, {**loss_dict, **encoder_statistics}
