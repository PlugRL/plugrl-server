import torch

from .dppo_config import SchedulerConfig
from .dppo_scheduler import NoOpScheduler
from .third_party.scheduler import CosineAnnealingWarmupRestarts



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
    # The schedule asserts this itself, with no message and no numbers. A run
    # shorter than its own warmup is a plausible mistake - it is what a first
    # smoke test of two iterations against the cheetah variant's ten-iteration
    # warmup does - and a bare AssertionError from inside a vendored file is a
    # bad way to find out.
    if scheduler_config.warmup_steps >= train_itrs:
        raise ValueError(
            f"warmup_steps ({scheduler_config.warmup_steps}) must be fewer "
            f"than the iterations being run ({train_itrs}). Either train for "
            f"longer or lower the scheduler's warmup."
        )
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=train_itrs,
        max_lr=max_lr,
        min_lr=scheduler_config.min_lr,
        warmup_steps=scheduler_config.warmup_steps,
    )
