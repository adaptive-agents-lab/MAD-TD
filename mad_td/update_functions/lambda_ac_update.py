import jax
import jax.numpy as jnp
from mad_td.update_functions.actor_updates import update_actor as actor_update_fn
from mad_td.update_functions.critic_updates import update_critic as critic_update_fn
from mad_td.update_functions.model_updates import update_model as model_update_fn
from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels

from mad_td.rl_types import (
    RLBatch,
    select_idx_batch,
)


def vmaped_utd_update(
    models,
    batch,
    hyperparams: MadTdHyperparams,
    shape,
    num_seeds,
    num_update_steps,
    key,
):
    return_dicts = []
    for i in range(num_update_steps):
        key, train_step_key = jax.random.split(key)
        train_step_key = jax.random.split(train_step_key, num_seeds)
        models, return_dict = jax.vmap(
            mad_td_update_step,
            in_axes=(
                shape,
                0,
                None,
                0,
                None,
                None,
                None,
            ),
        )(
            select_idx_batch(batch, i),
            models,
            hyperparams,
            train_step_key,
            True,
            i == 0 or not hyperparams.slow_actor,
            i == 0 or not hyperparams.slow_model,
        )
        return_dicts.append(return_dict)

    return models, return_dicts[0]


def mad_td_update_step(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
    update_critic: bool = True,
    update_actor: bool = True,
    update_model: bool = True,
):
    critic_key, actor_key, model_key = jax.random.split(key, 3)

    # batch = truncate_batch(batch, hyperparams.length_training_rollout)
    if update_model:
        models, model_loss_dict = model_update_fn(
            batch=batch,
            models=models,
            hyperparams=hyperparams,
            key=model_key,
        )
    else:
        model_loss_dict = {}
    if update_critic:
        models, critic_loss_dict = critic_update_fn(
            batch=batch,
            models=models,
            hyperparams=hyperparams,
            key=critic_key,
        )
    else:
        critic_loss_dict = {}
    if update_actor:
        models, actor_loss_dict = actor_update_fn(
            batch=batch,
            models=models,
            hyperparams=hyperparams,
            key=actor_key,
        )
    else:
        actor_loss_dict = {}
    # flatten across all batch dimensions in the logging dict
    info_dict = jax.tree_map(
        lambda x: jnp.mean(x),
        {
            "model": model_loss_dict,
            "critic": critic_loss_dict,
            "actor": actor_loss_dict,
        },
    )

    return models, info_dict
