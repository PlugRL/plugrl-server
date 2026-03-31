import dataclasses

from plugrl_server.algorithm.base_algorithm import BaseAlgoConfig
from plugrl_server.algorithm.registration import register_algo_config

UID = "fpo"


@register_algo_config(UID)
@dataclasses.dataclass
class FPOAlgoConfig(BaseAlgoConfig):
    global_steps: int | None = 60_000_000
    buffer_size: int = 2048
    discounting: float = 0.995
    reward_scaling: float = 10.0
    gae_lambda: float = 0.95
    batch_size: int = 256
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    value_loss_coeff: float = 0.25
    clipping_epsilon: float = 0.05
    normalize_advantage: bool = True
    n_samples_per_action: int = 8
    discretize_t_for_training: bool = True
    average_losses_before_exp: bool = True
    save_interval: int = 10
