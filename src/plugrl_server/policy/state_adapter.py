from __future__ import annotations

import dataclasses

import torch

from plugrl_server.common.data_utils import NumpyTree, TorchTree, numpy_tree_to_torch
from plugrl_server.policy.state import PolicyTrainState

TrainStateLike = PolicyTrainState


@dataclasses.dataclass(frozen=True)
class TrainStateTensors:
    obs: TorchTree
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


def train_state_to_tensors(
    train_state: TrainStateLike,
) -> TrainStateTensors:
    if train_state is None:
        raise ValueError("train_state must not be None")
    return TrainStateTensors(
        obs=_to_torch_tree(train_state["obs"]),
        action=_to_torch_tensor(train_state["action"]),
        logprob=_to_torch_tensor(train_state["logprob"]),
        entropy=_to_torch_tensor(train_state["entropy"]),
        value=_to_torch_tensor(train_state["value"]),
    )


def _to_torch_tree(value: NumpyTree) -> TorchTree:
    return numpy_tree_to_torch(value)


def _to_torch_tensor(value: NumpyTree) -> torch.Tensor:
    tensor = numpy_tree_to_torch(value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor leaf, got {type(tensor)!r}.")
    return tensor

