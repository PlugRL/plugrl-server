from __future__ import annotations

import dataclasses
from typing import Any

import tensordict
import torch

from plugrl_server.policy.base_policy import PolicyTensorState
from plugrl_server.policy.state import PolicyTrainState

TrainStateLike = PolicyTrainState | PolicyTensorState


@dataclasses.dataclass(frozen=True)
class TrainStateTensors:
    obs: torch.Tensor | tensordict.TensorDict
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def train_state_to_tensors(
    train_state: TrainStateLike,
) -> TrainStateTensors:
    if isinstance(train_state, PolicyTensorState):
        return TrainStateTensors(
            obs=train_state.obs,
            action=train_state.action,
            logprob=train_state.logprob,
            entropy=train_state.entropy,
            value=train_state.value,
        )
    if train_state is None:
        raise ValueError("train_state must not be None")
    return TrainStateTensors(
        obs=_to_torch_tree(train_state["obs"]),
        action=_to_torch_tree(train_state["action"]),
        logprob=_to_torch_tree(train_state["logprob"]),
        entropy=_to_torch_tree(train_state["entropy"]),
        value=_to_torch_tree(train_state["value"]),
    )


def _to_torch_tree(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {key: _to_torch_tree(item) for key, item in value.items()}
        batch_size = _infer_batch_size(converted)
        if batch_size is None:
            return converted
        return tensordict.TensorDict(converted, batch_size=batch_size)
    return torch.from_numpy(value)


def _infer_batch_size(value: dict[str, Any]) -> list[int] | None:
    for item in value.values():
        if isinstance(item, torch.Tensor):
            return list(item.shape[:1])
        if isinstance(item, tensordict.TensorDict):
            return list(item.batch_size)
    return None
