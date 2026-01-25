import dataclasses
import plugrl_server.algorithm.dppo.dppo_dist as _dppo_dist
from plugrl_server.algorithm.registration import register_algo_config
from plugrl_server.algorithm.dppo.dppo_config import SchedulerConfig


@register_algo_config(_dppo_dist.UID, "hopper")
@dataclasses.dataclass
class DPPOAlgoDistributedConfiHopper(_dppo_dist.DPPOAlgoDistributedConfig):
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


@register_algo_config(_dppo_dist.UID, "libero")
@dataclasses.dataclass
class DPPOAlgoConfigLibero(_dppo_dist.DPPOAlgoDistributedConfig):
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
