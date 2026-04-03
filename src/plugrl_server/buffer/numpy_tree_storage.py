from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from plugrl_server.policy.state import (
    ArraySpec,
    NumpyState,
    TrainStateSpec,
    infer_numpy_state_spec,
)


class NumpyTreeStorage:
    def __init__(self, spec: TrainStateSpec | ArraySpec, capacity: int):
        self.spec = spec
        self.capacity = capacity
        self.data = _allocate_from_spec(spec, capacity)

    @classmethod
    def from_example(cls, example: NumpyState, capacity: int) -> "NumpyTreeStorage":
        spec = infer_numpy_state_spec(example)
        return cls(spec=spec, capacity=capacity)

    def set_item(self, index: int | slice, value: NumpyState) -> None:
        _set_tree_item(self.data, index, value)

    def get_item(self, index: int | slice | np.ndarray) -> NumpyState:
        return _get_tree_item(self.data, index)

    def as_dict(self) -> dict:
        return dict(spec=self.spec, data=self.data)

    def load_dict(self, payload: dict) -> None:
        self.spec = payload["spec"]
        self.data = payload["data"]


def _allocate_from_spec(spec: TrainStateSpec | ArraySpec, capacity: int) -> NumpyState:
    if isinstance(spec, ArraySpec):
        return np.empty((capacity,) + spec.shape, dtype=np.dtype(spec.dtype))
    return dict(
        (key, _allocate_from_spec(value, capacity)) for key, value in spec.items()
    )


def _set_tree_item(tree: NumpyState, index: int | slice, value: NumpyState) -> None:
    if isinstance(tree, np.ndarray):
        assert isinstance(value, np.ndarray)
        tree[index] = value
        return
    assert isinstance(value, Mapping)
    for key, item in tree.items():
        _set_tree_item(item, index, value[key])


def _get_tree_item(tree: NumpyState, index: int | slice | np.ndarray) -> NumpyState:
    if isinstance(tree, np.ndarray):
        return tree[index]
    return dict((key, _get_tree_item(value, index)) for key, value in tree.items())
