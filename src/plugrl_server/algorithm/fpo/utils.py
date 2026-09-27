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
    """

    steps: int | None = None
    dims: int | None = None
    sum_over_steps: bool = False


def _reduce(
    error: torch.Tensor, batch_size: int, sample_count: int, reduction: ChunkReduction
) -> torch.Tensor:
    if reduction == ChunkReduction():
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
        return _reduce((v - target).pow(2), batch_size, sample_count, reduction)

    x0_pred = x_t - loss_t_expand * v
    x1_pred = x0_pred + v
    return _reduce((loss_eps - x1_pred).pow(2), batch_size, sample_count, reduction)
