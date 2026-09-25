import dataclasses
import pathlib

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
    n_samples_per_action: int = 8
    discretize_t_for_training: bool = True
    average_losses_before_exp: bool = True
    save_interval: int = 10
    # A checkpoint to start this run from, and how much of it to take.
    #
    # `load_checkpoint` has always restored model, optimizer and step count,
    # but nothing reached it: only `eval` had a flag, so a training run could
    # be saved and never resumed. E6 trained HalfCheetah for 500,000 steps and
    # its checkpoints could only be evaluated.
    #
    # `restore` exists because resuming and starting-from-weights are
    # different things, and the difference is what E14 ran into:
    #
    #   all           model, optimizer, step, iteration - a true resume.
    #   model         weights only. Optimizer, step and iteration start over.
    #   except-critic everything but `critic.*`, so the value head keeps its
    #                 random initialisation. This is E14's shape: pi0.5
    #                 arrived pretrained and carrying its own normalisation,
    #                 and only the value head was new.
    #
    # `except-critic` deliberately restores `obs_stats_*`, which sit outside
    # both `actor.` and `critic.`. Leaving them at initialisation would feed
    # the restored actor observations normalised differently from the ones it
    # was trained on, which is a second change, not the one being tested.
    policy_checkpoint_path: pathlib.Path | None = None
    restore: str = "all"
    # Where the float32 copies of half-precision parameters live, and with
    # them the optimizer's state. `None` keeps them beside the model. A second
    # card takes about 3.5 GB off the first for a pi0.5 action expert, which
    # is what a first learn step leaves behind and cannot give back. The cost
    # is a transfer per step; measure before assuming it is worth it.
    master_weights_device: str | None = None

    def __post_init__(self) -> None:
        allowed = ("all", "model", "except-critic")
        if self.restore not in allowed:
            raise ValueError(f"restore must be one of {allowed}, got {self.restore!r}")
        if self.restore != "all" and self.policy_checkpoint_path is None:
            raise ValueError(
                "restore only means something with policy_checkpoint_path set"
            )
