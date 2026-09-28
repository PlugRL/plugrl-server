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
    # How each Monte Carlo sample's squared error over an action chunk becomes
    # the loss its ratio compares. Unset, FPO averages every element of the
    # chunk. FPO++ (Yi, Choi et al. 2026; amazon-far/fpo-control), fine-tuning
    # pretrained flow policies, takes the mean over action dimensions and sums
    # it over the chunk's steps, all of which it executes.
    #
    # For pi0.5 on LIBERO the average ran over 10 steps of 32 dimensions; the
    # environment uses 7 and the client replans after 5, so 89% of it was
    # padding or actions never taken. E26's first iteration raised that
    # average from 0.0005 to 0.142 inside a clip of 0.05.
    cfm_loss_steps: int | None = None  # only a chunk's first this-many steps
    cfm_loss_dims: int | None = None  # only the first this-many action dims
    cfm_loss_sum_over_steps: bool = False
    # One ratio per Monte Carlo sample, each clipped on its own, with the
    # log-ratio clamped straight-through at +-5: FPO++'s per-sample ratio.
    # Off keeps one ratio per action, the samples averaged as
    # `average_losses_before_exp` says and the difference clamped at +-3.
    ratio_per_sample: bool = False
    # FPO++'s square fine-tuning (amazon-far/fpo-control, `manipulation_
    # experiments/finetune_online_rl.py`) has more than its policy loss, and
    # E37, which ran only the loss, watched a behaviour-cloned policy fall
    # while its randomly initialised critic never fit the returns. The rest,
    # each off by default:
    #
    # Two AdamW optimizers. FPO++ gives the actor 1e-5 with betas (0.9, 0.99)
    # and the critic 1e-4 with the default betas, both eps 1e-5 and weight
    # decay 1e-6. Here they are two parameter groups of one AdamW, built when
    # any of these differs from FPO's one Adam over both.
    critic_learning_rate: float | None = None
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    actor_adam_beta2: float = 0.999
    # Clip the actor's and the critic's gradients separately to this norm
    # (FPO++: 25 on square).
    max_grad_norm: float | None = None
    # Normalise the advantages within each minibatch rather than over the
    # buffer, as FPO++ does with its 375-chunk minibatches. See
    # `FPOAlgorithm._scale_advantage` for why the buffer is the default: at
    # the minibatches of 8 a VLA forces, a per-minibatch statistic
    # manufactures signal.
    normalize_advantage_per_minibatch: bool = False
    # FPO++'s Huber on the flow-matching error: d^2 within delta, 2 delta |d|
    # - delta^2 beyond (FPO++: delta 1), before the chunk reduction.
    cfm_loss_huber_delta: float | None = None
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
        if self.normalize_advantage_per_minibatch and not self.normalize_advantage:
            raise ValueError(
                "normalize_advantage_per_minibatch needs normalize_advantage on"
            )
