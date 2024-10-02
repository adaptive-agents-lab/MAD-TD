from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels

from mad_td.rl_types import RLBatch


def update_actor(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
) -> Tuple[MadTdModels, Dict]:
    state = batch.state
    batch_action = batch.action[:, 0]
    latent_state = models.encoder.apply_fn(models.encoder.params, state)

    def _j(actor_fw_params, state):
        # target real
        action = models.actor.apply_fn(actor_fw_params, state)

        target = models.critic_target.apply_fn(models.critic_target.params, state, action)
        target = jnp.stack(target, axis=0)
        target = jnp.mean(jnp.min(target, axis=0))

        return -jnp.mean(target), {
            "mean_action": jnp.mean(action),
            "mean_abs_action": jnp.mean(jnp.abs(action)),
            "action_churn": jnp.mean(jnp.square(action - batch_action)),
        }

    loss_grad = jax.value_and_grad(_j, argnums=0, has_aux=True, allow_int=True)
    (target, loss_dict), grad = loss_grad(
        models.actor.params,
        latent_state,
    )

    # update parameters

    loss_dict["actor_loss"] = jnp.mean(target)
    new_actor = models.actor.apply_gradients(grads=grad)
    new_models = MadTdModels(
        encoder=models.encoder,
        encoder_target=models.encoder_target,
        latent_model=models.latent_model,
        critic=models.critic,
        critic_target=models.critic_target,
        actor=new_actor,
    )

    return new_models, loss_dict
