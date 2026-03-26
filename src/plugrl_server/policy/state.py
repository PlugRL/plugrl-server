import dataclasses
from typing import Any, TypeAlias
from collections.abc import Mapping

import numpy as np
import torch
import tensordict

PolicyStateArray: TypeAlias = np.ndarray
NumpyState: TypeAlias = PolicyStateArray | Mapping[str, "NumpyState"]
PolicyRuntimeState: TypeAlias = Any | None
PolicyTrainState: TypeAlias = Mapping[str, NumpyState] | None


@dataclasses.dataclass
class PolicyStepState:
    runtime_state: PolicyRuntimeState = None
    train_state: PolicyTrainState = None


def slice_policy_step_state(step_state: PolicyStepState, index: Any) -> PolicyStepState:
    return PolicyStepState(
        runtime_state=slice_batched_state(step_state.runtime_state, index),
        train_state=slice_batched_state(step_state.train_state, index),
    )


def slice_batched_state(state: PolicyRuntimeState, index: Any) -> PolicyRuntimeState:
    if state is None:
        return None
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return type(state)(
            **dict(
                (field.name, slice_batched_state(getattr(state, field.name), index))
                for field in dataclasses.fields(state)
            )
        )
    if isinstance(state, dict):
        return {key: slice_batched_state(value, index) for key, value in state.items()}
    return state[index]


def is_numpy_state_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(item, np.ndarray) or is_numpy_state_mapping(item)
        for item in value.values()
    )


def to_numpy_state(value: Any) -> NumpyState | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return dict((key, to_numpy_state(item)) for key, item in value.items())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dict(
            (field.name, to_numpy_state(getattr(value, field.name)))
            for field in dataclasses.fields(value)
        )
    return _to_numpy_leaf(value)


def _to_numpy_leaf(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, tensordict.TensorDict):
        return dict((key, _to_numpy_leaf(item)) for key, item in value.items())
    if isinstance(value, dict):
        return dict((key, _to_numpy_leaf(item)) for key, item in value.items())
    return value
