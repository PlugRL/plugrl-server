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
    # Named as DPPO names it: `n_critic_warmup_itrs` is the field that
    # algorithm has had, and the README documents it. FPO had no equivalent.
    # The mechanism differs because the optimizers do - DPPO keeps separate
    # actor and critic optimizers and simply does not step the actor, while
    # FPO has one and zeroes the actor's gradients between backward and the
    # step - but the meaning is the same.
    #
    # FPO weights each policy update by an advantage, and an advantage is a
    # return minus the value head's estimate of it. That head starts from a
    # random initialisation, so on the first iteration the weights are noise.
    # E14 measured `losses/value_loss` at 0.880 on its first iteration,
    # falling to 9.3e-05 by its ninth: the critic could not predict returns at
    # all when it supplied the advantages for the first policy update, and the
    # policy went from 29 of 50 to 0 of 50 across that update.
    #
    # Tested on pi0.5 and refuted: one warmup iteration, actor frozen and
    # verified frozen by an evaluation at 31 of 50, then a real update that
    # returned 0 of 50. Kept because the capability is worth having and
    # because DPPO has it, not because it explains anything.
    #
    # Zero keeps the behaviour every experiment so far has run with.
    n_critic_warmup_itrs: int = 0
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
    # Tested on pi0.5 and refuted: 0.01, the best of five settings on that
    # bandit, returned 0 of 50 on both iterations, exactly as E14 did without
    # it. Kept as a detector and a capability, not as a fix.
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
