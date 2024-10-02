import functools
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np

from mad_td.cfgs import mad_td_config
from mad_td.cfgs.mad_td_config import MadTdModels
from mad_td.rl_types import RLBatch, select_idx_batch
from mad_td.update_functions.critic_updates import get_target_fn

@jax.jit
def get_action(obs, models):
    latent = models.encoder.apply_fn(models.encoder.params, obs)
    action = models.actor.apply_fn(models.actor.params, latent)
    return action


def get_batch_on_policy_tuples(batch: RLBatch, models, env):
    assert batch.physics_state is not None
    reset_fn = env.get_reset()
    step_fn = env.get_step(jax.random.PRNGKey(0))
    actions = jax.vmap(get_action, in_axes=(0, 0))(batch.state, models)
    rewards_ = []
    next_states_ = []
    for i in range(batch.state.shape[1]):
        keys = [jax.random.PRNGKey(0) for _ in range(batch.state.shape[0])]

        reset_dict = [
            {"internal_state": batch.physics_state[j, i]}
            for j in range(batch.physics_state.shape[0])
        ]
        env_state = reset_fn(keys, options=reset_dict)
        env_state, reward, done = step_fn(keys, env_state, actions[:, i])
        rewards_.append(reward)
        next_states_.append(env_state.obs)
    rewards = np.stack(rewards_, axis=1)
    next_states = np.stack(next_states_, axis=1)
    return RLBatch(
        state=batch.state,
        action=actions.reshape(*batch.action.shape),
        reward=rewards.reshape(*batch.reward.shape),
        next_state=next_states.reshape(*batch.next_state.shape),
        mask=batch.mask,
        idxs=batch.idxs,
        physics_state=batch.physics_state
    )


@functools.partial(jax.jit, static_argnums=(2))
def critic_loss(batch: RLBatch, models, hyperparams, key):
    loss_fn, target_fn = get_target_fn(hyperparams)

    state = batch.state
    action = batch.action[:, 0]
    reward = batch.reward[:, 0]
    next_state = batch.next_state[:, 0]
    mask = batch.mask[:, 0]

    loss_dict = {}

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
    (_, loss_dict) = loss_fn(
        latent_state,
        action,
        target,
        models.critic,
        mask,
        target_key,
        hyperparams,
    )
    return loss_dict


def eval_on_policy_batches(
    batches: RLBatch,
    models: MadTdModels,
    env,
    algo_cfg: mad_td_config.MadTdHyperparams,
    batch_shape,
):
    algo_cfg = algo_cfg.replace(proportion_real=1.0)

    buffer_loss_dicts = []
    on_policy_loss_dicts = []

    for i in tqdm(range(batches.state.shape[1])):
        buffer_batch = select_idx_batch(batches, i)
        on_policy_batch = get_batch_on_policy_tuples(
            buffer_batch, models, env
        )

        buffer_loss_dict = jax.vmap(critic_loss, in_axes=(batch_shape, 0, None, None))(
            buffer_batch, models, algo_cfg, jax.random.PRNGKey(0)
        )
        on_policy_loss_dict = jax.vmap(critic_loss, in_axes=(batch_shape, 0, None, None))(
            on_policy_batch, models, algo_cfg, jax.random.PRNGKey(0)
        )

        buffer_loss_dicts.append(buffer_loss_dict)
        on_policy_loss_dicts.append(on_policy_loss_dict)

    return [
        {f"eval_buffer/{k}": v for k, v in buffer_loss_dict.items()}
        for buffer_loss_dict in buffer_loss_dicts
    ], [
        {f"eval_on_policy/{k}": v for k, v in on_policy_loss_dict.items()}
        for on_policy_loss_dict in on_policy_loss_dicts
    ]
