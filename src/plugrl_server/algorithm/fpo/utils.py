from __future__ import annotations

from typing import Any

import torch

from plugrl_server.policy.base_policy_gradient_diffusion_policy import TorchTree
from plugrl_server.policy.base_policy_gradient_flow_policy import (
    BasePolicyGradientFlowPolicy,
)


def compute_cfm_loss(
    policy: BasePolicyGradientFlowPolicy,
    action: torch.Tensor,
    *,
    loss_eps: torch.Tensor,
    loss_t: torch.Tensor,
    processed_obs: Any,
) -> torch.Tensor:
    batch_size = action.shape[0]
    action_shape = action.shape[1:]
    sample_count = loss_eps.shape[1]
    loss_t_expand = loss_t.reshape(
        batch_size, sample_count, *([1] * len(action_shape))
    )
    x_t = loss_t_expand * loss_eps + (1.0 - loss_t_expand) * action.unsqueeze(1)

    x_t_flat = x_t.reshape(batch_size * sample_count, *action_shape)
    t_flat = loss_t.reshape(batch_size * sample_count)
    v = policy._predict_v(
        x_t_flat,
        t_flat,
        None,
        processed_cond=processed_obs,
    ).reshape_as(x_t)

    if getattr(policy, "output_mode", None) == "u":
        target = loss_eps - action.unsqueeze(1)
        return (v - target).pow(2).reshape(batch_size, sample_count, -1).mean(dim=-1)

    x0_pred = x_t - loss_t_expand * v
    x1_pred = x0_pred + v
    return (loss_eps - x1_pred).pow(2).reshape(batch_size, sample_count, -1).mean(dim=-1)
