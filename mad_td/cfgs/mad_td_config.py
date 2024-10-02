from dataclasses import dataclass, replace
from typing import Callable, Dict

from flax.training.train_state import TrainState
from flax import struct


@dataclass
class MadTdLossFunctions:
    loss_functions: Dict[str, Callable]

    def __hash__(self):
        return hash(self.loss_functions.keys())

    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

    def __repr__(self) -> str:
        return str(self.loss_functions.keys())


@dataclass
class MadTdHyperparams:
    batch_size: int

    actor_learning_rate: float
    critic_learning_rate: float
    encoder_learning_rate: float
    model_learning_rate: float
    gradient_clip: float
    gamma: float
    tau: float
    length_training_rollout: int
    td_lambda: float = 1.0
    quantile: bool = False
    c51: bool = False
    num_quantile: int = 151
    num_bins: int = 151
    is_binned: bool = False
    use_symlog: bool = False
    vmin: float = 0.0
    vmax: float = 200.0
    binned_regression_encoding: str = "two_hot"  # 'hl-gauss' or 'two-hot'
    update_encoder_with_critic_loss: bool = False
    skip_model_update: bool = False
    use_model_states_for_actor_update: bool = True
    proportion_real: float = 0.5

    use_resetting: bool = False
    reset_interval: int = 200000
    use_spr_reset: bool = False
    use_actor_reset: bool = False

    slow_actor: bool = False
    slow_model: bool = False
    average_critic_update: bool = False

    use_daml: bool = True
    use_model_loss: bool = True

    obs_scale: float = 1.0
    reward_scale: float = 1.0

    cross_kl_weight: float = 0.0

    byol_explore: bool = False
    byol_explore_weight: float = 100.0

    init_weight: float = 1.0

    # MPC parameters
    use_mpc: bool = False
    num_policy_mpc_samples: int = 1
    num_mpc_samples: int = 1
    mpc_iterations: int = 1
    mpc_top_k: int = 1
    mpc_temperature: float = 1.0
    use_cheap_mpc: bool = False

    use_target_encoder: bool = True
    use_model_gradient_for_actor: bool = True

    mpc_rollout_length: int = 1
    mpc_lower_std_bound: float = 0.05
    mpc_upper_std_bound: float = 0.3
    mpc_add_final_noise: bool = False

    mpc_v: str = "mean"

    def __hash__(self):
        return hash(
            (
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.encoder_learning_rate,
                self.model_learning_rate,
                self.gradient_clip,
                self.gamma,
                self.length_training_rollout,
                self.td_lambda,
                self.quantile,
                self.c51,
                self.num_quantile,
                self.vmin,
                self.vmax,
            )
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@struct.dataclass
class MadTdModels:
    critic: TrainState
    critic_target: TrainState
    actor: TrainState
    encoder: TrainState
    encoder_target: TrainState
    latent_model: TrainState
