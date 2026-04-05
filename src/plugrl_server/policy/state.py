import dataclasses
from typing import Any, TypeAlias, cast
from collections.abc import Mapping

import numpy as np
import torch
import tensordict

from plugrl_server.common.data_utils import torch_tensor_to_numpy

PolicyStateArray: TypeAlias = np.ndarray
NumpyState: TypeAlias = PolicyStateArray | Mapping[str, "NumpyState"]
PolicyRuntimeState: TypeAlias = Any | None
PolicyTrainState: TypeAlias = Mapping[str, NumpyState] | None
TrainStateSpec: TypeAlias = dict[str, "TrainStateSpec | ArraySpec"]


@dataclasses.dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: str


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
    return cast(Any, state)[index]


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
        return dict(
            (key, _require_numpy_state(to_numpy_state(item), key))
            for key, item in value.items()
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dict(
            (
                field.name,
                _require_numpy_state(
                    to_numpy_state(getattr(value, field.name)), field.name
                ),
            )
            for field in dataclasses.fields(value)
        )
    return _to_numpy_leaf(value)


def infer_train_state_spec(train_state: PolicyTrainState) -> TrainStateSpec | None:
    if train_state is None:
        return None
    return dict(
        (key, infer_numpy_state_spec(value)) for key, value in train_state.items()
    )


def _to_numpy_leaf(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return torch_tensor_to_numpy(value)
    if isinstance(value, tensordict.TensorDict):
        return dict((key, _to_numpy_leaf(item)) for key, item in value.items())
    if isinstance(value, dict):
        return dict((key, _to_numpy_leaf(item)) for key, item in value.items())
    return value


def _require_numpy_state(value: NumpyState | None, field_name: str) -> NumpyState:
    if value is None:
        raise TypeError(f"NumpyState field '{field_name}' cannot be None.")
    return value


def infer_numpy_state_spec(value: NumpyState) -> TrainStateSpec | ArraySpec:
    if isinstance(value, np.ndarray):
        array_value = cast(np.ndarray, value)
        sample_shape = tuple(array_value.shape[1:]) if array_value.ndim > 0 else tuple()
        return ArraySpec(shape=sample_shape, dtype=str(array_value.dtype))
    assert isinstance(value, Mapping), (
        f"Unsupported train_state leaf for spec inference: {type(value)!r}"
    )
    return dict((key, infer_numpy_state_spec(item)) for key, item in value.items())
