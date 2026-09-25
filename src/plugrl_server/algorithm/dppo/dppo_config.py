import dataclasses
import pathlib
from plugrl_server.algorithm.base_algorithm import BaseAlgoConfig
from plugrl_server.algorithm.registration import register_algo_config

UID = "dppo"


@dataclasses.dataclass
class SchedulerConfig:
    min_lr: float
    warmup_steps: int = 0


@register_algo_config(UID)
@dataclasses.dataclass
class DPPOAlgoConfig(BaseAlgoConfig):
    gamma: float = 0.99
    gamma_denoising: float = 1.0

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_weight_decay: float = 0.0
    critic_weight_decay: float = 0.0
    actor_lr_scheduler: SchedulerConfig | None = None
    critic_lr_scheduler: SchedulerConfig | None = None

    buffer_size: int = 20000
    gae_lambda: float = 0.95
    update_epochs: int = 4
    norm_adv: bool = True
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 0.001
    clip_ploss_coef_rate: float = 3
    clip_vloss_coef: float = float("inf")
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = float("inf")
    target_kl: float = float("inf")

    logprob_noise_level: float = 0.01
    sampling_noise_level: float = 0.01
    clip_advantage_lower_quantile: float = 0
    clip_advantage_upper_quantile: float = 1
    n_critic_warmup_itrs: int = 0
    use_normalized_rewards: bool = False

    batch_size: int = 256
    train_itrs: int = 200
    save_interval: int = 10
    grad_accum_steps: int = 8
    # A checkpoint to start from, and how much of it to take - the same three
    # modes FPO has, for the same reason: DPPO had a `load_checkpoint` that
    # nothing could reach, so a DPPO run could be saved and never continued,
    # and could not start from a policy some other run had trained.
    #
    #   all            model, both optimizers, step, iteration - a resume.
    #                  Needs a checkpoint DPPO wrote; FPO's has one optimizer.
    #   model          weights only; optimizers, step and iteration start over.
    #   except-critic  everything but `critic.*`. The value head keeps its
    #                  random initialisation and `obs_stats_*` come with the
    #                  actor. The mode to start DPPO from a policy FPO trained:
    #                  FPO's critic predicts returns under FPO's reward
    #                  scaling and discount, not DPPO's.
    #
    # Restored at the end of `init_optimizers`, not in `__init__`, because
    # `all` needs the optimizers to exist and DPPO builds them there.
    policy_checkpoint_path: pathlib.Path | None = None
    restore: str = "all"

    def __post_init__(self):
        allowed = ("all", "model", "except-critic")
        if self.restore not in allowed:
            raise ValueError(f"restore must be one of {allowed}, got {self.restore!r}")
        if self.restore != "all" and self.policy_checkpoint_path is None:
            raise ValueError(
                "restore only means something with policy_checkpoint_path set"
            )
        self.global_steps = self.train_itrs * self.buffer_size


@register_algo_config(UID, "hopper")
@dataclasses.dataclass
class DPPOAlgoConfigHopper(DPPOAlgoConfig):
    gamma: float = 0.99
    gamma_denoising: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_weight_decay: float = 0.0
    actor_lr_scheduler: SchedulerConfig | None = dataclasses.field(
        default_factory=lambda: SchedulerConfig(
            warmup_steps=10,
            min_lr=1e-4,
        )
    )
    critic_lr_scheduler: SchedulerConfig | None = dataclasses.field(
        default_factory=lambda: SchedulerConfig(
            warmup_steps=10,
            min_lr=1e-3,
        )
    )

    buffer_size: int = 40 * 500
    gae_lambda: float = 0.95

    batch_size: int = 2048
    update_epochs: int = 5
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 0.01
    logprob_noise_level: float = 0.1
    sampling_noise_level: float = 0.1
    use_normalized_rewards: bool = True


@register_algo_config(UID, "walker")
@dataclasses.dataclass
class DPPOAlgoConfigWalker(DPPOAlgoConfigHopper):
    pass


@register_algo_config(UID, "cheetah")
@dataclasses.dataclass
class DPPOAlgoConfigCheetah(DPPOAlgoConfigHopper):
    pass


@register_algo_config(UID, "libero")
@dataclasses.dataclass
class DPPOAlgoConfigLibero(DPPOAlgoConfig):
    gamma: float = 0.999
    gamma_denoising: float = 1.0
    actor_lr: float = 5e-6
    critic_lr: float = 1e-4
    actor_weight_decay: float = 0.0
    actor_lr_scheduler: SchedulerConfig | None = None
    critic_lr_scheduler: SchedulerConfig | None = None

    buffer_size: int = 64 * 32 * 8
    gae_lambda: float = 0.95

    batch_size: int = 128
    grad_accum_steps: int = 16
    update_epochs: int = 4
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_ploss_coef: float = 0.001
    clip_ploss_coef_base: float = 0.001

    logprob_noise_level: float = 0.5
    sampling_noise_level: float = 0.5
    use_normalized_rewards: bool = False

    save_interval: int = 2
