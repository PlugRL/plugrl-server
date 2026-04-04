import dataclasses
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
    critic_batch_size: int | None = None
    train_itrs: int = 200
    save_interval: int = 10
    grad_accum_steps: int = 8

    def __post_init__(self):
        if self.critic_batch_size is None:
            self.critic_batch_size = self.batch_size
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

    n_train_itr: int = 1000
    batch_size: int = 2048
    update_epochs: int = 5
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 0.01
    n_critic_warmup_itr: int = 0
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

    n_train_itr: int = 1000
    batch_size: int = 128
    grad_accum_steps: int = 16
    update_epochs: int = 4
    vf_coef: float = 0.5
    norm_adv: bool = True
    clip_ploss_coef: float = 0.001
    clip_ploss_coef_base: float = 0.001
    n_critic_warmup_itr: int = 0

    logprob_noise_level: float = 0.5
    sampling_noise_level: float = 0.5
    use_normalized_rewards: bool = False

    save_interval: int = 2
