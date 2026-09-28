from __future__ import annotations

import dataclasses

import torch

from plugrl_server.policy.base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
)


@dataclasses.dataclass(frozen=True)
class ChunkReduction:
    """How one Monte Carlo sample's squared error over an action chunk becomes a loss.

    The default averages every element of the chunk, which is what FPO has
    always done. `steps` and `dims` keep only a chunk's first steps and first
    action dimensions - the ones the client executes and the environment
    uses - and `sum_over_steps` sums the per-step means instead of averaging
    them, as FPO++ does (Yi, Choi et al. 2026, amazon-far/fpo-control).
    `huber_delta` replaces each element's squared error with FPO++'s Huber:
    d^2 within delta, 2 delta |d| - delta^2 beyond, which meets d^2 at delta.
    """

    steps: int | None = None
    dims: int | None = None
    sum_over_steps: bool = False
    huber_delta: float | None = None


def _elementwise_error(diff: torch.Tensor, huber_delta: float | None) -> torch.Tensor:
    if huber_delta is None:
        return diff.pow(2)
    magnitude = diff.abs()
    return torch.where(
        magnitude <= huber_delta,
        diff.pow(2),
        2.0 * huber_delta * magnitude - huber_delta**2,
    )


def _reduce(
    error: torch.Tensor, batch_size: int, sample_count: int, reduction: ChunkReduction
) -> torch.Tensor:
    if (
        reduction.steps is None
        and reduction.dims is None
        and not reduction.sum_over_steps
    ):
        return error.reshape(batch_size, sample_count, -1).mean(dim=-1)
    if error.dim() != 4:
        raise ValueError(
            f"a chunk reduction needs actions shaped (steps, dims), got {tuple(error.shape[2:])}"
        )
    steps, dims = error.shape[2:]
    if reduction.steps is not None and not 0 < reduction.steps <= steps:
        raise ValueError(
            f"cfm_loss_steps={reduction.steps}, but a chunk has {steps} steps"
        )
    if reduction.dims is not None and not 0 < reduction.dims <= dims:
        raise ValueError(
            f"cfm_loss_dims={reduction.dims}, but an action has {dims} dims"
        )
    per_step = error[:, :, : reduction.steps, : reduction.dims].mean(dim=-1)
    return per_step.sum(dim=-1) if reduction.sum_over_steps else per_step.mean(dim=-1)


def compute_cfm_loss(
    policy: BasePolicyGradientFlowPolicy,
    obs,
    action: torch.Tensor,
    *,
    output_mode: str,
    loss_eps: torch.Tensor,
    loss_t: torch.Tensor,
    obs_cache=None,
    reduction: ChunkReduction = ChunkReduction(),
) -> torch.Tensor:
    batch_size = action.shape[0]
    action_shape = action.shape[1:]
    sample_count = loss_eps.shape[1]
    loss_t_expand = loss_t.reshape(batch_size, sample_count, *([1] * len(action_shape)))
    x_t = loss_t_expand * loss_eps + (1.0 - loss_t_expand) * action.unsqueeze(1)

    x_t_flat = x_t.reshape(batch_size * sample_count, *action_shape)
    t_flat = loss_t.reshape(batch_size * sample_count)
    v = policy._predict_v(
        x_t_flat,
        t_flat,
        obs,
        cond_cache=obs_cache,
    ).reshape_as(x_t)

    if output_mode == "u":
        target = loss_eps - action.unsqueeze(1)
        error = _elementwise_error(v - target, reduction.huber_delta)
        return _reduce(error, batch_size, sample_count, reduction)

    x0_pred = x_t - loss_t_expand * v
    x1_pred = x0_pred + v
    error = _elementwise_error(loss_eps - x1_pred, reduction.huber_delta)
    return _reduce(error, batch_size, sample_count, reduction)
