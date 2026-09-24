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
    # Iterations during which only the value head learns and the policy is
    # left alone.
    #
    # FPO weights each policy update by an advantage, and an advantage is a
    # return minus the value head's estimate of it. That head starts from a
    # random initialisation, so on the first iteration the weights are noise.
    # E14 measured `losses/value_loss` at 0.880 on its first iteration,
    # falling to 9.3e-05 by its ninth: the critic could not predict returns at
    # all when it supplied the advantages for the first policy update, and the
    # policy went from 29 of 50 to 0 of 50 across that update.
    #
    # Whether the two facts are connected is what this exists to test. Zero
    # keeps the behaviour every experiment so far has run with.
    critic_warmup_iterations: int = 0
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
