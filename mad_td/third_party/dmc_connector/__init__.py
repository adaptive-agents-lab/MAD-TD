from math import inf
from typing import SupportsFloat, Any

import gymnasium
import gymnasium as gym
import numpy as np
from gym.core import ActType
from gymnasium.core import ObsType, WrapperObsType, WrapperActType
from gymnasium.envs.registration import register

from gymnasium import spaces, Env


def make(
    domain_name,
    task_name,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=False,
    action_noise=False,
    action_noise_type="normal",
    action_noise_level=0.0,
    noise_distraction: bool = False,
    obs_distortion: bool = False,
    num_distractions: int = 0,
):
    env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.envs.registry.keys():
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        register(
            id=env_id,
            entry_point="mad_td.third_party.dmc_connector.wrappers:DMCWrapper",
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
                action_noise=action_noise,
                action_noise_type=action_noise_type,
                action_noise_level=action_noise_level,
            ),
            max_episode_steps=max_episode_steps,
        )
    env = gym.make(env_id)
    if num_distractions > 0:
        return DistractionWrapper(env, seed)
    return env


def vector_make(
    domain_name,
    task_name,
    num_envs,
    seeds,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
    action_noise=False,
    action_noise_type="normal",
    action_noise_level=0.0,
    noise_distraction: bool = False,
    obs_distortion: bool = False,
    num_distractions: int = 0,
):
    assert (
        len(seeds) == num_envs or len(seeds) == 1
    ), "seeds must be either of length 1 or equal to num_envs"
    if len(seeds) == 1:
        seeds = seeds * num_envs
    ids = []
    for i in range(num_envs):
        env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seeds[i])
        ids.append(env_id)

        if from_pixels:
            assert (
                not visualize_reward
            ), "cannot use visualize reward when learning from pixels"

        # shorten episode length
        max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

        if env_id not in gym.envs.registry.keys():
            task_kwargs = {}
            if seeds[i] is not None:
                task_kwargs["random"] = seeds[i]
            if time_limit is not None:
                task_kwargs["time_limit"] = time_limit
            register(
                id=env_id,
                entry_point="mad_td.third_party.dmc_connector.wrappers:DMCWrapper",
                kwargs=dict(
                    domain_name=domain_name,
                    task_name=task_name,
                    task_kwargs=task_kwargs,
                    environment_kwargs=environment_kwargs,
                    visualize_reward=visualize_reward,
                    from_pixels=from_pixels,
                    height=height,
                    width=width,
                    camera_id=camera_id,
                    frame_skip=frame_skip,
                    channels_first=channels_first,
                    action_noise=action_noise,
                    action_noise_type=action_noise_type,
                    action_noise_level=action_noise_level,
                ),
                max_episode_steps=max_episode_steps,
            )

    # this is vital due to closure rules in python, DO NOT SIMPLIFY
    def make_env_func(id, seed):
        return lambda: DistractionWrapper(
            env=gym.make(id),
            seed=seed,
            noise_distraction=noise_distraction,
            obs_distortion=obs_distortion,
            num_distractions=num_distractions,
        )

    return gym.vector.AsyncVectorEnv(
        [make_env_func(id, seed) for (id, seed) in zip(ids, seeds)]
    )


class DistractionWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        seed: int = 1,
        noise_distraction: bool = False,
        obs_distortion: bool = False,
        num_distractions: int = 1,
    ):
        super().__init__(env)
        self.noise_distraction = noise_distraction
        self.obs_distortion = obs_distortion
        self.num_distractions = num_distractions
        self.seed = seed
        self.distruction_envs = [
            make(
                "humanoid",
                "run",
                seed=seed,
            )
            for _ in range(num_distractions)
        ]

    @property
    def observation_space(self) -> spaces.Space[ObsType] | spaces.Space[WrapperObsType]:
        super_space = super().observation_space
        space_shape = super_space.shape[0]
        for distraction_env in self.distruction_envs:
            space_shape += distraction_env.observation_space.shape[0]
        return spaces.Box(low=-inf, high=inf, shape=(space_shape,))

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        ob, rew, ter, tru, info = super().step(action)
        obs = [ob]
        for distraction_env in self.distruction_envs:
            distraction_ob, _, te, tr, _ = distraction_env.step(
                distraction_env.action_space.sample()
            )
            if te or tr:
                distraction_ob, _ = distraction_env.reset(seed=self.seed)
            if self.noise_distraction:
                distraction_ob = np.random.randn(*distraction_ob.shape)
            obs.append(distraction_ob)
        ob = np.concatenate(obs, axis=0)
        return ob, rew, ter, tru, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        ob, info = super().reset(seed=seed, options=options)
        obs = [ob]
        for distraction_env in self.distruction_envs:
            distraction_ob, _ = distraction_env.reset(seed=seed)
            obs.append(distraction_ob)
        ob = np.concatenate(obs, axis=0)
        return ob, info
