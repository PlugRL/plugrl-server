import dataclasses
from .dppo import DPPOAlgoConfig, UID, SchedulerConfig
from ..registration import register_algo_config

@register_algo_config(UID, "hopper")
@dataclasses.dataclass
class DPPOAlgoConfigHopper(DPPOAlgoConfig):
    gamma: float = 0.99
    gamma_denoising: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_weight_decay: float = 0.0
    actor_lr_scheduler: SchedulerConfig = dataclasses.field(
        default_factory=lambda: SchedulerConfig(
            warmup_steps=10,
            min_lr=1e-4,
        )
    )
    critic_lr_scheduler: SchedulerConfig = dataclasses.field(
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