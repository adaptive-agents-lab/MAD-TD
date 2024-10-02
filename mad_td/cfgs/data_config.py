from dataclasses import dataclass


@dataclass
class EnvConfig:
    name: str
    state_shape: int


@dataclass
class BraxEnvConfig(EnvConfig):
    backend: str = "positional"


@dataclass
class GymnaxEnvConfig(EnvConfig):
    pass


@dataclass
class DMCEnvConfig(EnvConfig):
    domain_name: str
    task_name: str
    seed: int
    action_noise: bool
    action_noise_type: str
    action_noise_level: float
    num_envs: int
    frame_skip: int
    obs_distortion: bool
    num_distractions: int
    noise_distraction: bool


@dataclass
class ManiSkillEnvConfig(EnvConfig):
    num_envs: int


@dataclass
class MyoSuiteEnvConfig(EnvConfig):
    num_envs: int
