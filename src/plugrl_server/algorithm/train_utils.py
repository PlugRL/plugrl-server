from collections.abc import Iterable

import torch

from plugrl_server.common.data_utils import torch_tree_to_device


def move_batch_to_device(
    batch: tuple,
    *,
    device: torch.device,
) -> tuple:
    obs, *rest = batch
    return (
        torch_tree_to_device(obs, device),
        *(tensor.to(device) for tensor in rest),
    )


def optimizer_step_if_ready(
    *,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    accum_steps: int,
    grad_accum_steps: int,
    max_grad_norm: float = float("inf"),
    force: bool = False,
    accumulated: int | None = None,
) -> float | None:
    """Step, if this is the end of an accumulation window.

    `accumulated` is how many backward passes have gone into the gradients
    since the last step. When given, the gradients are divided by it here, so
    the optimizer sees the *mean* minibatch gradient however many accumulated.

    That division used to happen at the loss instead - each minibatch's loss
    divided by `grad_accum_steps` - which is only correct when a full window
    accumulates. DPPO's caller flushes at the end of every epoch and breaks
    early on `target_kl`, so partial windows are the normal case, not the edge
    one: E17 ran with `grad_accum_steps` 8 and two minibatches per epoch and
    applied a quarter of the gradient it meant to, for 100 iterations.

    Left optional so a caller that has already scaled its loss is unaffected.
    """
    if not force and accum_steps % grad_accum_steps != 0:
        return None

    parameters = list(parameters)
    if accumulated is not None and accumulated > 1:
        scale = 1.0 / accumulated
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf")).item()
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return grad_norm
