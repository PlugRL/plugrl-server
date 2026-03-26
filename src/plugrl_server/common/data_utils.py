from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

BatchDict: TypeAlias = dict[str, "BatchValue"]
BatchValue: TypeAlias = BatchDict | np.ndarray | list[Any]
TorchTree: TypeAlias = torch.Tensor | dict[str, "TorchTree"]
NumpyTree: TypeAlias = np.ndarray | dict[str, "NumpyTree"]


def batch_aggregate(
    list_of_dicts: list[dict[str, Any]],
    aggregate_method: Literal["stack", "concat"] = "stack",
) -> BatchDict:
    if not list_of_dicts:
        return dict()

    keys = list_of_dicts[0].keys()
    result: BatchDict = dict()

    for key in keys:
        values = [d[key] for d in list_of_dicts]
        first_value = values[0]

        if isinstance(first_value, np.ndarray):
            try:
                if aggregate_method == "stack":
                    result[key] = np.stack(values, axis=0)
                elif aggregate_method == "concat":
                    result[key] = np.concatenate(values, axis=0)
                else:
                    raise ValueError(
                        f"Unsupported aggregate_method: {aggregate_method}"
                    )
            except ValueError as e:
                print(
                    f"Warning: Incompatible np.ndarray shapes under key '{key}' ({e}), aggregating as list."
                )
                result[key] = values

        elif isinstance(first_value, dict):
            if all(isinstance(v, dict) for v in values):
                result[key] = batch_aggregate(values, aggregate_method)
            else:
                result[key] = values

        else:
            result[key] = values

    return result


def unbatch_aggregate(
    batched_dict: BatchDict, aggregate_method: Literal["stack", "concat"] = "stack"
) -> list[dict[str, Any]]:
    if not batched_dict:
        return []

    keys = batched_dict.keys()
    batch_size = None

    for key in keys:
        value = batched_dict[key]
        if isinstance(value, np.ndarray):
            batch_size = value.shape[0]
            break
        if isinstance(value, dict):
            nested_values = list(value.values())
            for nested_value in nested_values:
                if isinstance(nested_value, np.ndarray):
                    batch_size = nested_value.shape[0]
                    break
        if batch_size is not None:
            break

    if batch_size is None:
        return [batched_dict]

    result: list[dict[str, Any]] = [dict() for _ in range(batch_size)]

    for key in keys:
        value = batched_dict[key]

        if isinstance(value, np.ndarray):
            for i in range(batch_size):
                if aggregate_method == "stack":
                    result[i][key] = value[i]
                elif aggregate_method == "concat":
                    result[i][key] = value[i : i + 1]
                else:
                    raise ValueError(
                        f"Unsupported aggregate_method: {aggregate_method}"
                    )
        elif isinstance(value, dict):
            nested_unbatched = unbatch_aggregate(value, aggregate_method)
            for i, nested_dict in enumerate(nested_unbatched):
                result[i][key] = nested_dict
        else:
            for i in range(batch_size):
                result[i][key] = value[i]

    return result


def create_empty_torch_tree(template: TorchTree, buffer_size: int) -> TorchTree:
    if isinstance(template, torch.Tensor):
        return torch.empty(
            (buffer_size,) + tuple(template.shape),
            dtype=template.dtype,
            device=template.device,
        )
    return dict(
        (key, create_empty_torch_tree(value, buffer_size))
        for key, value in template.items()
    )


def torch_tree_get_item(tree: TorchTree, index: int | slice | torch.Tensor) -> TorchTree:
    if isinstance(tree, torch.Tensor):
        return tree[index]
    return dict((key, torch_tree_get_item(value, index)) for key, value in tree.items())


def torch_tree_set_item(tree: TorchTree, index: int | slice, value: TorchTree) -> None:
    if isinstance(tree, torch.Tensor):
        assert isinstance(value, torch.Tensor)
        tree[index] = value
        return
    assert isinstance(value, Mapping)
    for key, item in tree.items():
        torch_tree_set_item(item, index, value[key])


def stack_torch_tree(items: Sequence[TorchTree], dim: int = 0) -> TorchTree:
    if not items:
        raise ValueError("Cannot stack an empty torch tree sequence.")
    first_item = items[0]
    if isinstance(first_item, torch.Tensor):
        return torch.stack(list(items), dim=dim)
    return dict(
        (key, stack_torch_tree([item[key] for item in items], dim=dim))
        for key in first_item.keys()
    )


def stack_numpy_tree(items: Sequence[NumpyTree], axis: int = 0) -> NumpyTree:
    if not items:
        raise ValueError("Cannot stack an empty numpy tree sequence.")
    first_item = items[0]
    if isinstance(first_item, np.ndarray):
        return np.stack(list(items), axis=axis)
    return dict(
        (key, stack_numpy_tree([item[key] for item in items], axis=axis))
        for key in first_item.keys()
    )


def torch_tree_to_numpy(tree: TorchTree) -> NumpyTree:
    if isinstance(tree, torch.Tensor):
        return tree.cpu().numpy()
    return dict((key, torch_tree_to_numpy(value)) for key, value in tree.items())


def numpy_tree_to_torch(tree: NumpyTree) -> TorchTree:
    if isinstance(tree, np.ndarray):
        return torch.from_numpy(tree)
    return dict((key, numpy_tree_to_torch(value)) for key, value in tree.items())


def torch_tree_to_device(tree: TorchTree, device: torch.device) -> TorchTree:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    return dict((key, torch_tree_to_device(value, device)) for key, value in tree.items())
