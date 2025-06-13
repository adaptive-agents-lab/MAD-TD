import abc
from dataclasses import dataclass
import functools

from typing import Any, Callable, Sequence, Tuple

import numpy as onp
import gymnax
from brax import envs
import jax
import jax.numpy as jnp
from jax import tree_util
from flax import struct
import gymnasium

import mad_td.third_party.dmc_connector as dmc2gym
from mad_td.cfgs.data_config import (
    BraxEnvConfig,
    DMCEnvConfig,
    EnvConfig,
    GymnaxEnvConfig,
    ManiSkillEnvConfig,
)


@struct.dataclass
class EnvState:
    obs: jax.Array
    state: Any


@dataclass
class Env(abc.ABC):
    config: EnvConfig

    @abc.abstractmethod
    def get_reset(self) -> Callable[[jax.Array], EnvState]:
        pass

    @abc.abstractmethod
    def get_n_reset(self) -> Callable[[jax.Array], EnvState]:
        pass

    @abc.abstractmethod
    def get_step(
        self,
    ) -> Callable[
        [jax.Array, jax.Array, jax.Array], Tuple[EnvState, jax.Array, jax.Array]
    ]:
        pass

    @abc.abstractmethod
    def get_n_step(
        self, key: jax.Array
    ) -> Callable[
        [jax.Array, jax.Array, jax.Array], Tuple[EnvState, jax.Array, jax.Array]
    ]:
        pass

    @abc.abstractmethod
    def get_observation_space(self) -> Sequence[int]:
        pass

    @abc.abstractmethod
    def get_action_space(self) -> Sequence[int]:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


@dataclass
class GymnaxEnv(Env):
    config: GymnaxEnvConfig

    def __post_init__(self):
        self.env, self.env_params = gymnax.make(self.config.name)

    def get_reset(self):
        # obs, state = env.reset(key_reset, env_params)
        def _reset(key, env_params):
            obs, state = self.env.reset(key, env_params)
            return EnvState(obs, state)

        return jax.jit(functools.partial(_reset, env_params=self.env_params))

    def get_n_reset(self):
        reset = jax.vmap(self.get_reset(), in_axes=(0))
        return jax.jit(reset)

    def get_step(self):
        # n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
        def _step(key, state, action, env_params):
            n_obs, n_state, reward, done, _ = self.env.step(
                key, state, action, env_params
            )
            return EnvState(n_obs, n_state), reward, done

        return jax.jit(functools.partial(_step, env_params=self.env_params))

    def get_n_step(self):
        step = jax.vmap(self.get_step(), in_axes=(0, 0, 0))
        return jax.jit(step)

    def get_observation_space(self):
        return self.env.observation_space(self.env_params).shape

    def get_action_space(self):
        return self.env.action_space(self.env_params).shape

    def close(self):
        self.env.close()


@dataclass
class BraxEnv(Env):
    config: BraxEnvConfig

    def __post_init__(self):
        self.env = envs.get_environment(self.config.name, backend=self.config.backend)

    def get_reset(self):
        def _reset(key):
            state = self.env.reset(key)
            return EnvState(state.obs, state)

        return jax.jit(_reset)

    def get_n_reset(self):
        return jax.vmap(self.get_reset(), in_axes=(0))

    def get_step(self):
        def _step(key, state, action):
            state = self.env.step(state, action)
            return EnvState(state.obs, state), state.reward, state.done

        return jax.jit(_step)

    def get_n_step(self):
        def _step(key, state, action):
            state_shape = tree_util.tree_map(lambda x: 0, state)
            return jax.vmap(self.get_step(), in_axes=(None, state_shape, 0))(
                key, state, action
            )

        return jax.jit(_step)

    def get_observation_space(self):
        return (self.env.observation_size,)

    def get_action_space(self):
        return (self.env.action_size,)


@dataclass
class DMCEnv(Env):
    config: DMCEnvConfig

    def __post_init__(self):
        self.vec_env = dmc2gym.vector_make(
            domain_name=self.config.domain_name,
            task_name=self.config.task_name,
            num_envs=self.config.num_envs,
            seeds=[self.config.seed + i for i in range(self.config.num_envs)],
            frame_skip=self.config.frame_skip,
            action_noise=self.config.action_noise,
            action_noise_level=self.config.action_noise_level,
            obs_distortion=self.config.obs_distortion,
            num_distractions=self.config.num_distractions,
            noise_distraction=self.config.noise_distraction,
        )

    def get_reset(self):
        def _reset(keys, options=None):
            # unsafe casting is used to ensure positive key, this should be harmless
            keys = onp.array(keys, dtype=onp.uint32)[:, 0].tolist()
            # else:
            #     keys = onp.array(keys, dtype=onp.uint32)[0].item()
            obs, info = self.vec_env.reset(seed=keys, options=options)
            obs = jnp.array(obs)
            return EnvState(obs, info)

        return _reset

    def get_n_reset(self):
        return self.get_reset()

    def get_step(self, key):
        def _step(key, state, action):
            action = onp.array(action)
            action = onp.clip(action, -1, 1)
            obs, reward, truncated, terminated, info = self.vec_env.step(action)
            if "final_observation" in info:
                obs = onp.stack(info["final_observation"], axis=0)
            obs = obs
            reward = jnp.array(reward)
            truncated = jnp.array(truncated)
            terminated = jnp.array(terminated)
            return EnvState(obs, info), reward, jnp.logical_or(truncated, terminated)

        return _step

    def get_n_step(self, key):
        return self.get_step(key)

    def get_observation_space(self):
        return self.vec_env.observation_space.shape[1:]

    def get_action_space(self):
        return self.vec_env.action_space.shape[1:]

    def close(self):
        self.vec_env.close()


@dataclass
class ManiSkillEnv(Env):
    config: ManiSkillEnvConfig

    def __post_init__(self):
        self.env = gymnasium.make(
            self.config.name,
            num_envs=self.config.num_envs,
            obs_mode="state",  # there is also "state_dict", "rgbd", ...
            control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
        )

    def get_reset(self):
        def _reset(keys):
            keys = onp.array(keys, dtype=onp.uint32)[:, 0].tolist()
            obs, _ = self.env.reset(seed=keys[0])
            obs = jnp.array(obs.cpu())
            return EnvState(obs, None)

        return _reset

    def get_n_reset(self):
        return self.get_reset()

    def get_step(self, key):
        def _step(key, state, action):
            action = onp.array(action)
            action = onp.clip(action, -1, 1)
            obs, reward, truncated, terminated, info = self.env.step(action)
            if "final_observation" in info:
                obs = onp.stack(info["final_observation"], axis=0)
            obs = jnp.array(obs.cpu())
            reward = jnp.array(reward.cpu())
            truncated = jnp.array(truncated.cpu())
            if isinstance(terminated, bool):
                terminated = jnp.array(terminated)
            else:
                terminated = jnp.array(terminated.cpu())
            return EnvState(obs, None), reward, jnp.logical_or(truncated, terminated)

        return _step

    def get_n_step(self, key):
        return self.get_step(key)

    def get_observation_space(self):
        return self.env.observation_space.shape

    def get_action_space(self):
        return self.env.action_space.shape


def make_env(env_config: EnvConfig) -> Env:
    if isinstance(env_config, GymnaxEnvConfig):
        env = GymnaxEnv(env_config)
    elif isinstance(env_config, BraxEnvConfig):
        env = BraxEnv(env_config)
    elif isinstance(env_config, DMCEnvConfig):
        env = DMCEnv(env_config)
    elif isinstance(env_config, ManiSkillEnvConfig):
        env = ManiSkillEnv(env_config)
    else:
        raise ValueError(f"Unknown env config {env_config}")
    return env
