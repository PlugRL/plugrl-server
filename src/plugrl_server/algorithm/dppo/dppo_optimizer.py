import torch

from .dppo_config import SchedulerConfig
from .dppo_scheduler import NoOpScheduler

try:
    import dppo.util.scheduler as _dppo_scheduler
except ImportError:
    raise ImportError(
        'dppo is not installed. Please install it with pip install "plugrl-server[dppo]".'
    )


def build_adamw(
    parameters,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_config: SchedulerConfig | None,
    train_itrs: int,
    max_lr: float,
):
    if scheduler_config is None:
        return NoOpScheduler()
    return _dppo_scheduler.CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=train_itrs,
        max_lr=max_lr,
        min_lr=scheduler_config.min_lr,
        warmup_steps=scheduler_config.warmup_steps,
    )
