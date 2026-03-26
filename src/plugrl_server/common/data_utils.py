from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)

BatchDict: TypeAlias = dict[str, "BatchValue"]
BatchValue: TypeAlias = BatchDict | np.ndarray | list[Any]
TorchTree: TypeAlias = torch.Tensor | dict[str, "TorchTree"]
NumpyTree: TypeAlias = np.ndarray | Mapping[str, "NumpyTree"]


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
                logger.warning(
                    "Incompatible ndarray shapes under key '%s' (%s), aggregating as list.",
                    key,
                    e,
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
            value_array = cast(np.ndarray, value)
            batch_size = value_array.shape[0]
            break
        if isinstance(value, Mapping):
            nested_values = list(value.values())
            for nested_value in nested_values:
                if isinstance(nested_value, np.ndarray):
                    nested_array = cast(np.ndarray, nested_value)
                    batch_size = nested_array.shape[0]
                    break
        if batch_size is not None:
            break

    if batch_size is None:
        return [batched_dict]

    result: list[dict[str, Any]] = [dict() for _ in range(batch_size)]

    for key in keys:
        value = batched_dict[key]

        if isinstance(value, np.ndarray):
            value_array = cast(np.ndarray, value)
            for i in range(batch_size):
                if aggregate_method == "stack":
                    result[i][key] = value_array[i]
                elif aggregate_method == "concat":
                    result[i][key] = value_array[i : i + 1]
                else:
                    raise ValueError(
                        f"Unsupported aggregate_method: {aggregate_method}"
                    )
        elif isinstance(value, Mapping):
            nested_unbatched = unbatch_aggregate(dict(value), aggregate_method)
            for i, nested_dict in enumerate(nested_unbatched):
                result[i][key] = nested_dict
        else:
            value_seq = cast(Sequence[Any], value)
            for i in range(batch_size):
                result[i][key] = value_seq[i]

    return result


def create_empty_torch_tree(template: TorchTree, buffer_size: int) -> TorchTree:
    if isinstance(template, torch.Tensor):
        tensor_template = cast(torch.Tensor, template)
        return torch.empty(
            (buffer_size,) + tuple(tensor_template.shape),
            dtype=tensor_template.dtype,
            device=tensor_template.device,
        )
    template_mapping = cast(Mapping[str, TorchTree], template)
    return dict(
        (key, create_empty_torch_tree(value, buffer_size))
        for key, value in template_mapping.items()
    )


def torch_tree_get_item(tree: TorchTree, index: int | slice | torch.Tensor) -> TorchTree:
    if isinstance(tree, torch.Tensor):
        tensor_tree = cast(torch.Tensor, tree)
        return tensor_tree[index]
    tree_mapping = cast(Mapping[str, TorchTree], tree)
    return dict(
        (key, torch_tree_get_item(value, index)) for key, value in tree_mapping.items()
    )


def torch_tree_set_item(tree: TorchTree, index: int | slice, value: TorchTree) -> None:
    if isinstance(tree, torch.Tensor):
        assert isinstance(value, torch.Tensor)
        tensor_tree = cast(torch.Tensor, tree)
        tensor_tree[index] = value
        return
    tree_mapping = cast(Mapping[str, TorchTree], tree)
    assert isinstance(value, Mapping)
    for key, item in tree_mapping.items():
        torch_tree_set_item(item, index, value[key])


def stack_torch_tree(items: Sequence[TorchTree], dim: int = 0) -> TorchTree:
    if not items:
        raise ValueError("Cannot stack an empty torch tree sequence.")
    first_item = items[0]
    if isinstance(first_item, torch.Tensor):
        tensor_items = cast(Sequence[torch.Tensor], items)
        return torch.stack(list(tensor_items), dim=dim)
    first_mapping = cast(Mapping[str, TorchTree], first_item)
    return dict(
        (
            key,
            stack_torch_tree(
                [cast(Mapping[str, TorchTree], item)[key] for item in items], dim=dim
            ),
        )
        for key in first_mapping.keys()
    )


def stack_numpy_tree(items: Sequence[NumpyTree], axis: int = 0) -> NumpyTree:
    if not items:
        raise ValueError("Cannot stack an empty numpy tree sequence.")
    first_item = items[0]
    if isinstance(first_item, np.ndarray):
        array_items = cast(Sequence[np.ndarray], items)
        return np.stack(list(array_items), axis=axis)
    first_mapping = cast(Mapping[str, NumpyTree], first_item)
    return dict(
        (
            key,
            stack_numpy_tree(
                [cast(Mapping[str, NumpyTree], item)[key] for item in items], axis=axis
            ),
        )
        for key in first_mapping.keys()
    )


def torch_tree_to_numpy(tree: TorchTree) -> NumpyTree:
    if isinstance(tree, torch.Tensor):
        tensor_tree = cast(torch.Tensor, tree)
        return tensor_tree.cpu().numpy()
    tree_mapping = cast(Mapping[str, TorchTree], tree)
    return dict((key, torch_tree_to_numpy(value)) for key, value in tree_mapping.items())


def numpy_tree_to_torch(tree: NumpyTree) -> TorchTree:
    if isinstance(tree, np.ndarray):
        array_tree = cast(np.ndarray, tree)
        return torch.from_numpy(array_tree)
    tree_mapping = cast(Mapping[str, NumpyTree], tree)
    return dict((key, numpy_tree_to_torch(value)) for key, value in tree_mapping.items())


def torch_tree_to_device(tree: TorchTree, device: torch.device) -> TorchTree:
    if isinstance(tree, torch.Tensor):
        tensor_tree = cast(torch.Tensor, tree)
        return tensor_tree.to(device)
    tree_mapping = cast(Mapping[str, TorchTree], tree)
    return dict(
        (key, torch_tree_to_device(value, device)) for key, value in tree_mapping.items()
    )
