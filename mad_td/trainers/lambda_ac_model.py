from dataclasses import dataclass

import jax
import optax
from flax.training import train_state
from mad_td.cfgs.mad_td_config import MadTdModels

from mad_td.data.replay_buffer import ReplayBuffer
from mad_td.rl_types import (
    AbstractActor,
    AbstractCritic,
    AbstractEncoder,
    AbstractLatentModel,
)


@dataclass
class MadTdFactory:
    critic: AbstractCritic
    critic_target: AbstractCritic
    actor: AbstractActor
    encoder: AbstractEncoder
    encoder_target: AbstractEncoder
    latent_model: AbstractLatentModel
    replay_buffer: ReplayBuffer
    encoder_learning_rate: float
    model_learning_rate: float
    critic_learning_rate: float
    actor_learning_rate: float
    gradient_clip: float
    seed: int

    def get_networks(self) -> MadTdModels:
        return MadTdModels(
            critic=self.critic_state,
            critic_target=self.target_state,
            actor=self.actor_state,
            encoder=self.encoder_state,
            encoder_target=self.encoder_target_state,
            latent_model=self.latent_model_state,
        )

    def init(self) -> MadTdModels:
        self.key = jax.random.PRNGKey(self.seed)
        self.key, init_key = jax.random.split(self.key)

        sample_batch = self.replay_buffer.get_dummy_batch()
        dummy_state = sample_batch.state
        dummy_action = sample_batch.action
        num_seeds = self.replay_buffer.num_seeds

        # initialize all models
        (
            critic_key,
            actor_key,
            encoder_key,
            latent_model_key,
        ) = jax.random.split(init_key, 4)

        def _create_train_states(keys):
            # split the keys
            critic_key = keys[0]
            actor_key = keys[1]
            encoder_key = keys[2]
            latent_model_key = keys[3]

            encoder_params = self.encoder.init(encoder_key, dummy_state)
            encoder_target_params = self.encoder_target.init(encoder_key, dummy_state)

            dummy_latent_state = self.encoder.apply(encoder_params, dummy_state)

            critic_params = self.critic.init(
                critic_key, dummy_latent_state, dummy_action
            )
            # print(self.critic.tabulate(critic_key, dummy_latent_state, dummy_action))

            target_params = self.critic_target.init(
                critic_key, dummy_latent_state, dummy_action
            )

            actor_params = self.actor.init(actor_key, dummy_latent_state)
            # print(self.actor.tabulate(actor_key, dummy_latent_state))
            latent_model_params = self.latent_model.init(
                latent_model_key, dummy_latent_state, dummy_action
            )
            # print(
            #     self.latent_model.tabulate(
            #         latent_model_key, dummy_latent_state, dummy_action
            #     )
            # )

            # initialize optimizers
            def _add_clip(optimizer):
                return optax.chain(
                    optax.clip_by_global_norm(self.gradient_clip), optimizer
                )

            critic_optim = _add_clip(
                optax.adam(learning_rate=self.critic_learning_rate)
            )
            actor_optim = _add_clip(optax.adam(learning_rate=self.actor_learning_rate))
            encoder_optim = _add_clip(
                optax.adam(learning_rate=self.encoder_learning_rate)
            )
            latent_model_optim = _add_clip(
                optax.adam(learning_rate=self.model_learning_rate)
            )

            # create train states
            critic_train_state = train_state.TrainState.create(
                apply_fn=self.critic.apply,
                params=critic_params,
                tx=critic_optim,
            )
            target_train_state = train_state.TrainState.create(
                apply_fn=self.critic.apply,
                params=target_params,
                tx=optax.set_to_zero(),
            )
            actor_train_state = train_state.TrainState.create(
                apply_fn=self.actor.apply, params=actor_params, tx=actor_optim
            )
            encoder_train_state = train_state.TrainState.create(
                apply_fn=self.encoder.apply, params=encoder_params, tx=encoder_optim
            )
            encoder_target_train_state = train_state.TrainState.create(
                apply_fn=self.encoder_target.apply,
                params=encoder_target_params,
                tx=optax.set_to_zero(),
            )
            latent_model_train_state = train_state.TrainState.create(
                apply_fn=self.latent_model.apply,
                params=latent_model_params,
                tx=latent_model_optim,
            )

            return (
                critic_train_state,
                target_train_state,
                actor_train_state,
                encoder_train_state,
                encoder_target_train_state,
                latent_model_train_state,
            )

        critic_key = jax.random.split(critic_key, num_seeds)
        actor_key = jax.random.split(actor_key, num_seeds)
        encoder_key = jax.random.split(encoder_key, num_seeds)
        latent_model_key = jax.random.split(latent_model_key, num_seeds)

        (
            self.critic_state,
            self.target_state,
            self.actor_state,
            self.encoder_state,
            self.encoder_target_state,
            self.latent_model_state,
        ) = jax.vmap(_create_train_states)(
            (
                critic_key,
                actor_key,
                encoder_key,
                latent_model_key,
            )
        )
        self.seed += 1

        return self.get_networks()
