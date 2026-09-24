import dataclasses

from plugrl_server.algorithm.base_algorithm import BaseAlgoConfig
from plugrl_server.algorithm.registration import register_algo_config

UID = "fpo"


@register_algo_config(UID)
@dataclasses.dataclass
class FPOAlgoConfig(BaseAlgoConfig):
    global_steps: int | None = 60_000_000
    buffer_size: int = 983_040
    output_mode: str = "u_but_supervise_as_eps"
    fpo_playground_trick: bool = True
    treat_truncated_as_done: bool = False
    discounting: float = 0.995
    reward_scaling: float = 10.0
    gae_lambda: float = 0.95
    batch_size: int = 1024
    num_updates_per_batch: int = 16
    learning_rate: float = 3e-4
    value_loss_coeff: float = 0.25
    clipping_epsilon: float = 0.05
    normalize_advantage: bool = True
    # Stop an iteration's updates once the policy has drifted this far from
    # the one that collected the data, measured as |1 - mean policy ratio|.
    #
    # Clipping bounds what a single sample can contribute; it does not bound
    # where thousands of updates end up, and PPO implementations pair it with
    # a stop for that reason. This one had no such stop.
    #
    # Measured on a bandit small enough to run on a CPU: FPO reaches about
    # -0.06 within twenty iterations and then loses it on four of ten
    # seed-and-dtype runs, by up to nine times. Over the same stretch the
    # correlation between the advantages and the reward they encode falls from
    # about 0.5 to 0.07 - the updates stop carrying information and do not
    # stop being applied.
    #
    # Zero disables it, which is what every run before this used.
    max_policy_drift: float = 0.0
    n_samples_per_action: int = 8
    discretize_t_for_training: bool = True
    average_losses_before_exp: bool = True
    save_interval: int = 10
    # Where the float32 copies of half-precision parameters live, and with
    # them the optimizer's state. `None` keeps them beside the model. A second
    # card takes about 3.5 GB off the first for a pi0.5 action expert, which
    # is what a first learn step leaves behind and cannot give back. The cost
    # is a transfer per step; measure before assuming it is worth it.
    master_weights_device: str | None = None
