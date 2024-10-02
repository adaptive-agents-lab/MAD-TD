import random
from gymnasium import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np


def _spec_to_box(spec):
    def extract_min_max(s):
        # assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int32(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    # assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True,
        action_noise=False,
        action_noise_type="normal",
        action_noise_level=0.0,
    ):
        assert (
            "random" in task_kwargs
        ), "please specify a seed, for deterministic behaviour"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        self._action_noise = action_noise
        self._action_noise_type = action_noise_type
        self._action_noise_level = action_noise_level

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        if self._action_noise:
            if self._action_noise_type == "normal":
                self._action_noise_distribution = lambda x: np.random.normal(
                    x, self._action_noise_level
                )
            elif self._action_noise_type == "uniform":
                self._action_noise_distribution = lambda x: np.random.uniform(
                    x - self._action_noise_level, x + self._action_noise_level
                )
            elif self._action_noise_type == "bimodal":
                self._action_noise_distribution = lambda x: [
                    np.random.normal(
                        x - self._action_noise_level, self._action_noise_level / 2.0
                    ),
                    np.random.normal(
                        x + self._action_noise_level, self._action_noise_level / 2.0
                    ),
                ][random.randint(0, 1)]
            else:
                raise NotImplementedError("Unknown action noise type")

        else:
            self._action_noise_distribution = lambda x: x

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_space = _spec_to_box(self._env.observation_spec().values())
        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get("random", 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, observation):
        if self._from_pixels:
            obs = self.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed=None):
        if seed is None:
            seed = self._seed + 1
        self._seed = seed
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        # assert self._norm_action_space.contains(action), (
        #     "action %s is not in action space" % action
        # )
        action = self._convert_action(action)
        # assert self._true_action_space.contains(action), (
        #     "action %s is not in true action space" % action
        # )
        reward = 0

        if self._action_noise:
            action = self._action_noise_distribution(action)

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        extra = {"internal_state": self._env.physics.get_state().copy()}
        obs = self._get_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, False, done, extra

    def reset(self, seed=None, options=None):
        self.seed(seed)
        time_step = self._env.reset()
        observation = time_step.observation
        if isinstance(options, dict) and "internal_state" in options:
            self._env.physics.set_state(options["internal_state"])
            self._env.physics.forward()
            observation = self._env.task.get_observation(self._env._physics)
        obs = self._get_obs(observation)
        state = self._env.physics.get_state().copy()
        self.current_state = _flatten_obs(observation)
        return obs, {"internal_state": state}

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        # assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
