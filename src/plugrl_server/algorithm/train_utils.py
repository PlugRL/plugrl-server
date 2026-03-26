from collections.abc import Iterable

import torch


def optimizer_step_if_ready(
    *,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    accum_steps: int,
    grad_accum_steps: int,
    max_grad_norm: float = float("inf"),
    force: bool = False,
) -> float | None:
    if not force and accum_steps % grad_accum_steps != 0:
        return None

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf")).item()
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return grad_norm
