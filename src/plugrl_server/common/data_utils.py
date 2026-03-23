import torch
import numpy as np
import tensordict
from typing import List, Dict, Any, Union, Literal

BatchDict = Dict[str, Union["BatchDict", np.ndarray, List[Any]]]


def batch_aggregate(
    list_of_dicts: List[Dict[str, Any]],
    aggregate_method: Literal["stack", "concat"] = "stack",
) -> BatchDict:
    if not list_of_dicts:
        return {}

    keys = list_of_dicts[0].keys()
    result = {}

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
                result[key] = batch_aggregate(values)
            else:
                result[key] = values

        else:
            result[key] = values

    return result


def unbatch_aggregate(
    batched_dict: BatchDict, aggregate_method: Literal["stack", "concat"] = "stack"
) -> List[Dict[str, Any]]:
    if not batched_dict:
        return []

    keys = batched_dict.keys()
    batch_size = None

    for key in keys:
        value = batched_dict[key]
        if isinstance(value, np.ndarray):
            batch_size = value.shape[0]
            break
        elif isinstance(value, dict):
            nested_values = list(value.values())
            for nv in nested_values:
                if isinstance(nv, np.ndarray):
                    batch_size = nv.shape[0]
                    break
        if batch_size is not None:
            break

    if batch_size is None:
        return [batched_dict]

    result = [{} for _ in range(batch_size)]

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


def _recursively_create_empty_td(template_td: torch.Tensor, buffer_size):
    new_data = {}
    for key, item in template_td.items():
        if isinstance(item, torch.Tensor):
            new_shape = (buffer_size,) + item.shape
            new_data[key] = torch.empty(new_shape, dtype=item.dtype, device=item.device)
        elif isinstance(item, tensordict.TensorDict):
            new_data[key] = _recursively_create_empty_td(item, buffer_size)
        else:
            new_data[key] = item

    return tensordict.TensorDict(
        new_data, batch_size=[buffer_size] + list(template_td.shape)
    )
