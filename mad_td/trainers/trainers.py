import os

import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from tqdm import tqdm

from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels

from mad_td.cfgs.train_config import TrainHyperparams
from mad_td.rl_util.rl_targets import soft_target_update
from mad_td.rl_types import (
    AbstractActor,
    AbstractCritic,
    AbstractEncoder,
    AbstractLatentModel,
    get_batch_seed_shape,
)
from mad_td.data.env import Env, make_env
from mad_td.data.replay_buffer import ReplayBuffer
from mad_td.utils.checkpointing import CheckpointHandler
from mad_td.utils import eval_utils
from mad_td.utils.jax import tree_list_mean
from mad_td.utils.adversarial import compute_all_adversarial_metrics
from mad_td.trainers.mad_td_model import (
    MadTdFactory,
)
from mad_td.update_functions.mad_td_update import vmaped_utd_update
from mad_td.utils.logging import (
    Logger,
    multi_seed_return_dict,
)


class MultiSeedTrainer:
    def __init__(
        self,
        critic: AbstractCritic,
        critic_target: AbstractCritic,
        actor: AbstractActor,
        encoder: AbstractEncoder,
        encoder_target: AbstractEncoder,
        latent_model: AbstractLatentModel,
        replay_buffer: ReplayBuffer,
        eval_replay_buffer: ReplayBuffer,
        env: Env,
        algo_hyperparams: MadTdHyperparams,
        train_hyperparams: TrainHyperparams,
        logger: Logger,
    ):
        self.mad_td = MadTdFactory(
            critic=critic,
            critic_target=critic_target,
            actor=actor,
            encoder=encoder,
            encoder_target=encoder_target,
            latent_model=latent_model,
            replay_buffer=replay_buffer,
            encoder_learning_rate=algo_hyperparams.encoder_learning_rate,
            model_learning_rate=algo_hyperparams.model_learning_rate,
            critic_learning_rate=algo_hyperparams.critic_learning_rate,
            actor_learning_rate=algo_hyperparams.actor_learning_rate,
            gradient_clip=algo_hyperparams.gradient_clip,
            seed=train_hyperparams.seed,
        )
        self.models = self.mad_td.init()

        self.encoder = encoder
        self.latent_model = latent_model
        self.critic = critic
        self.actor = actor

        # setup replay buffer
        self.replay_buffer = replay_buffer
        self.eval_replay_buffer = eval_replay_buffer

        # setup shapes for vmap
        self.batch_shape = get_batch_seed_shape(
            self.replay_buffer.get_dummy_batch(1, batch_size=10)
        )

        # setup hyperparams
        self.algo_hyperparams = algo_hyperparams
        self.train_hyperparams = train_hyperparams

        # setup utility
        self.key = jax.random.PRNGKey(train_hyperparams.seed)
        self.key, env_step_key = jax.random.split(self.key)

        # setup env
        self.env = env
        self.env_reset = self.env.get_n_reset()
        self.env_step = self.env.get_n_step(env_step_key)

        self.eval_env = make_env(env.config)

        self.logger = logger

        # prejit update step
        self.dummy_batch = self.replay_buffer.get_dummy_batch(
            num_samples=train_hyperparams.update_steps,
            batch_size=train_hyperparams.batch_size,
        )
        self._update_fn = (
            jax.jit(vmaped_utd_update, static_argnums=(2, 3, 4, 5))
            .lower(
                self.models,
                self.dummy_batch,
                algo_hyperparams,
                self.batch_shape,
                train_hyperparams.num_seeds,
                train_hyperparams.update_steps,
                self.key,
            )
            .compile()
        )

        # handle reloading logic here
        self.steps_done = 0

        self.checkpointer = CheckpointHandler(train_hyperparams.save_path)

    def check_pretrain(self, path, alt_path):
        if alt_path:
            path = os.path.join(alt_path, path)

        if os.path.exists(os.path.join(path, "steps_done.txt")):
            self.load(path)
            with open(os.path.join(path, "steps_done.txt"), "r") as f:
                self.steps_done = int(f.read())
            print(f"Resuming from previous checkpoint at {self.steps_done}")
        else:
            print(
                f"No checkpoint found at {os.path.join(path, 'steps_done.txt')}, starting from scratch"
            )

    def save(self, path):
        self.replay_buffer.save(path)
        self.eval_replay_buffer.save(path)
        self.checkpointer.checkpoint_params(self.models, self.steps_done)
        with open(os.path.join(path, "steps_done.txt"), "w") as f:
            f.write(str(self.steps_done))

    def load(self, path):
        self.replay_buffer.load(path)
        self.eval_replay_buffer.load(path)
        reload_checkpointer = CheckpointHandler(path)
        self.models = reload_checkpointer.restore_params(self.models, path)

    def reset_if_time(self):
        if (
            self.algo_hyperparams.use_resetting
            and (self.steps_done % self.algo_hyperparams.reset_interval) == 0
        ) and (self.steps_done != 0):
            print("Resetting models")
            new_models = self.mad_td.init()
            if self.algo_hyperparams.use_spr_reset:
                vmaped_target_update = jax.vmap(
                    soft_target_update, in_axes=(0, 0, None)
                )
                self.models = MadTdModels(
                    encoder=vmaped_target_update(
                        new_models.encoder, self.models.encoder, 0.5
                    ),
                    encoder_target=vmaped_target_update(
                        new_models.encoder_target,
                        self.models.encoder_target,
                        0.5,
                    ),
                    latent_model=vmaped_target_update(
                        new_models.latent_model, self.models.latent_model, 0.5
                    ),
                    critic=new_models.critic,
                    critic_target=new_models.critic_target,
                    actor=new_models.actor,
                )
            elif self.algo_hyperparams.use_actor_reset:
                self.models = MadTdModels(
                    encoder=self.models.encoder,
                    encoder_target=self.models.encoder_target,
                    latent_model=self.models.latent_model,
                    critic=self.models.critic,
                    critic_target=self.models.critic_target,
                    actor=new_models.actor,
                )
            else:
                self.models = MadTdModels(
                    encoder=new_models.encoder,
                    encoder_target=new_models.encoder_target,
                    latent_model=new_models.latent_model,
                    critic=new_models.critic,
                    critic_target=new_models.critic_target,
                    actor=new_models.actor,
                )
            self._update_fn = (
                jax.jit(vmaped_utd_update, static_argnums=(2, 3, 4, 5))
                .lower(
                    self.models,
                    self.dummy_batch,
                    self.algo_hyperparams,
                    self.batch_shape,
                    self.train_hyperparams.num_seeds,
                    self.train_hyperparams.update_steps,
                    self.key,
                )
                .compile()
            )

    def train(self):
        num_seeds = self.train_hyperparams.num_seeds
        key, reset_key = jax.random.split(self.key)
        done = False
        rewards = 0
        reset_rng = jax.random.split(reset_key, num_seeds)
        state = self.env_reset(reset_rng)

        return_dicts = []
        action_info_dicts = []

        total_steps = self.train_hyperparams.total_steps

        if self.algo_hyperparams.use_mpc:
            act_fn = get_mpc_action
        else:
            act_fn = get_policy_action
        vmaped_action = jax.jit(
            jax.vmap(
                lambda s, e, a, c, m, noise_sigma, random, key: act_fn(
                    s,
                    e,
                    a,
                    c,
                    m,
                    noise_sigma,
                    random,
                    self.algo_hyperparams,
                    key,
                ),
                in_axes=(
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    0,
                ),
            ),
            static_argnames=("random"),
        )

        with tqdm(total=total_steps) as pbar:
            pbar.update(self.steps_done)
            while self.steps_done <= total_steps:
                self.reset_if_time()
                for _ in range(self.train_hyperparams.env_steps):
                    # key logic
                    key, step_key = jax.random.split(key)
                    step_keys = jax.random.split(step_key, num_seeds)
                    if np.any(done):
                        state = self.env_reset(step_keys)
                        done = False
                        self.logger.log(
                            multi_seed_return_dict(
                                {"reward": rewards},
                                self.train_hyperparams.num_seeds,
                            ),
                            self.steps_done,
                        )
                        self.logger.log(
                            {"mean_reward": float(np.mean(rewards))},
                            self.steps_done,
                        )
                        rewards = 0
                    key, action_key = jax.random.split(key)
                    action_keys = jax.random.split(action_key, num_seeds)
                    action, action_info = vmaped_action(
                        state.obs * self.algo_hyperparams.obs_scale,
                        self.models.encoder,
                        self.models.actor,
                        self.models.critic,
                        self.models.latent_model,
                        0.3 if self.algo_hyperparams.use_mpc else 0.1,
                        self.steps_done < self.train_hyperparams.init_steps,
                        action_keys,
                    )
                    action_info_dicts.append(action_info)
                    next_state, reward, done = self.env_step(
                        step_keys, state.state, action
                    )
                    rewards += reward
                    replay_buffer_choice = np.random.choice([True, False], p=[1-self.train_hyperparams.eval_sample_ratio, self.train_hyperparams.eval_sample_ratio])
                    if replay_buffer_choice:
                        self.replay_buffer.insert(
                            state.obs * self.algo_hyperparams.obs_scale,
                            action,
                            reward * self.algo_hyperparams.reward_scale,
                            done,
                            next_state.obs * self.algo_hyperparams.obs_scale,
                        )
                    else:
                        self.eval_replay_buffer.insert(
                            state.obs * self.algo_hyperparams.obs_scale,
                            action,
                            reward * self.algo_hyperparams.reward_scale,
                            done,
                            next_state.obs * self.algo_hyperparams.obs_scale,
                        )
                    state = next_state
                self.steps_done += self.train_hyperparams.env_steps
                if self.steps_done > self.train_hyperparams.init_steps:
                    # split all required keys
                    key, model_key = jax.random.split(key)
                    # sample batches
                    batch = self.replay_buffer.sample(
                        self.train_hyperparams.batch_size, model_key, self.train_hyperparams.update_steps  # type: ignore
                    )
                    # update models
                    self.models, return_dict = self._update_fn(
                        self.models,
                        batch,
                        key,
                    )
                    return_dicts.append(return_dict)

                    # logging
                    if self.steps_done % self.train_hyperparams.log_freq == 0:
                        self.logger.log(
                            multi_seed_return_dict(
                                action_info, self.train_hyperparams.num_seeds
                            ),
                            self.steps_done,
                        )
                        action_info_dicts = tree_list_mean(action_info_dicts)
                        action_info_dicts = multi_seed_return_dict(
                            action_info_dicts, self.train_hyperparams.num_seeds
                        )
                        self.logger.log(action_info_dicts, self.steps_done)
                        del action_info_dicts
                        action_info_dicts = []
                        return_dict = tree_list_mean(return_dicts)
                        return_dict = multi_seed_return_dict(
                            return_dict, self.train_hyperparams.num_seeds
                        )
                        self.logger.log(return_dict, self.steps_done)
                        del return_dicts
                        return_dicts = []

                    if self.steps_done % 500 == 0:
                        self.logger.flush()
                    if self.steps_done % self.train_hyperparams.save_freq == 0:
                        self.save(self.train_hyperparams.save_path)
                if self.steps_done % self.train_hyperparams.tqdm_interval == 0:
                    pbar.update(
                        self.train_hyperparams.tqdm_interval
                        * self.train_hyperparams.env_steps
                    )


def get_policy_action(
    state: jax.Array,
    encoder: TrainState,
    actor: TrainState,
    critic: TrainState,
    model: TrainState,
    exploration_noise_sigma: jax.Array,
    random: bool,
    hyperparams: MadTdHyperparams,
    rand_key: jax.Array,
):
    feature = encoder.apply_fn(encoder.params, state)
    action = actor.apply_fn(actor.params, feature)

    exploration_noise = (
        jax.random.normal(rand_key, shape=action.shape) * exploration_noise_sigma
    )
    exploration_noise = jax.numpy.clip(exploration_noise, -0.3, 0.3)
    action = action + exploration_noise
    action = jax.numpy.clip(action, -1.0, 1.0)

    if random:
        action = jax.random.uniform(rand_key, shape=action.shape, minval=-1, maxval=1)
    # assert jax.numpy.all(jax.numpy.isfinite(action))
    return action, {}


def get_mpc_action(
    state: jax.Array,
    encoder: TrainState,
    actor: TrainState,
    critic: TrainState,
    model: TrainState,
    exploration_noise_sigma: jax.Array,
    random: bool,
    hyperparams: MadTdHyperparams,
    rand_key: jax.Array,
):
    z = encoder.apply_fn(encoder.params, state)

    if random:
        dummy_action = actor.apply_fn(actor.params, z)
        action = jax.random.uniform(
            rand_key, shape=dummy_action.shape, minval=-1, maxval=1
        )
        top_k_softmax = jnp.ones(hyperparams.mpc_top_k) / hyperparams.mpc_top_k
        info = {
            "mpc/action_diff": jnp.mean(jnp.square(action - dummy_action)),
            "mpc/chosen_action_diff": jnp.mean(jnp.square(action - dummy_action)),
            **{f"mpc/std_{i}": 0.0 for i in range(hyperparams.mpc_iterations)},
            **{
                f"mpc/softmax_entropy_{i}": -jnp.sum(
                    top_k_softmax * jnp.log(top_k_softmax + 1e-8)
                )
                for i in range(hyperparams.mpc_iterations)
            },
        }
        return action, info

    policy_action = actor.apply_fn(actor.params, z)

    z = z[jnp.newaxis, ...]
    z = jnp.repeat(z, hyperparams.num_mpc_samples + 1, axis=0)

    mean = policy_action
    std = jnp.ones_like(mean) * exploration_noise_sigma
    softmax_per_round = []
    stds_per_round = []
    for _ in range(hyperparams.mpc_iterations):
        traj_key, rand_key = jax.random.split(rand_key)

        random_actions = (
            mean
            + jax.random.normal(
                traj_key, shape=[hyperparams.num_mpc_samples, *policy_action.shape]
            )
            * std
        )

        actions = jnp.concatenate([policy_action[jnp.newaxis], random_actions], axis=0)
        actions = jnp.clip(actions, -1, 1)
        if hyperparams.use_cheap_mpc:
            values1, values2 = critic.apply_fn(critic.params, z, actions)
            values = (values1 + values2) / 2
        else:
            rewards = 0.0
            next_state_latent = z
            next_actions = actions
            for i in range(hyperparams.mpc_rollout_length):
                next_state_latent, reward = model.apply_fn(
                    model.params, next_state_latent, next_actions
                )
                rewards += reward * hyperparams.gamma**i
                next_actions = actor.apply_fn(actor.params, next_state_latent)
            q1, q2 = critic.apply_fn(critic.params, next_state_latent, next_actions)
            if hyperparams.mpc_v == "max":
                next_q = jnp.max(jnp.stack([q1, q2], axis=0), axis=0)
            elif hyperparams.mpc_v == "min":
                next_q = jnp.min(jnp.stack([q1, q2], axis=0), axis=0)
            elif hyperparams.mpc_v == "mean":
                next_q = (q1 + q2) / 2
            values = (
                rewards + hyperparams.gamma**hyperparams.mpc_rollout_length * next_q
            )

        elite_values, elite_idx = jax.lax.top_k(values.squeeze(), hyperparams.mpc_top_k)
        elite_weights = jax.nn.softmax(
            elite_values / hyperparams.mpc_temperature, axis=0
        )[:, jnp.newaxis]

        mean = jnp.sum(elite_weights * actions[elite_idx], axis=0, keepdims=True)
        std = jnp.sqrt(
            jnp.sum(
                elite_weights * jnp.square(actions[elite_idx] - mean),
                axis=0,
                keepdims=True,
            )
        )
        std = jnp.clip(
            std,
            hyperparams.mpc_lower_std_bound,
            hyperparams.mpc_upper_std_bound,
        )
        stds_per_round.append(jnp.mean(std))

        top_k_softmax = jax.nn.softmax(
            elite_values / hyperparams.mpc_temperature, axis=0
        )
        top = jax.random.choice(rand_key, elite_idx, p=top_k_softmax)
        softmax_per_round.append(
            -jnp.sum(top_k_softmax * jnp.log(top_k_softmax + 1e-8))
        )
    if hyperparams.mpc_add_final_noise:
        action = actions[top] + std[0] * jax.random.normal(
            rand_key, shape=actions[top].shape
        )
    else:
        action = actions[top]

    print(f"tok_k_softmax shape: {top_k_softmax.shape}")
    info = {
        "mpc/action_diff": jnp.mean(jnp.square(action - policy_action)),
        "mpc/chosen_action_diff": jnp.mean(jnp.square(actions[top] - policy_action)),
        **{f"mpc/std_{i}": std for i, std in enumerate(stds_per_round)},
        **{f"mpc/softmax_entropy_{i}": ent for i, ent in enumerate(softmax_per_round)},
    }

    return action, info
