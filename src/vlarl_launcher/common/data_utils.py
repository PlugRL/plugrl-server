import numpy as np
from typing import List, Dict, Any, Union

BatchDict = Dict[str, Union['BatchDict', np.ndarray, List[Any]]]

def batch_aggregate(list_of_dicts: List[Dict[str, Any]]) -> BatchDict:
    if not list_of_dicts:
        return {}

    keys = list_of_dicts[0].keys()
    result = {}

    for key in keys:
        values = [d[key] for d in list_of_dicts]
        first_value = values[0]

        if isinstance(first_value, np.ndarray):
            try:
                result[key] = np.concatenate(values, axis=0)
            except ValueError as e:
                print(f"Warning: Incompatible np.ndarray shapes under key '{key}' ({e}), aggregating as list.")
                result[key] = values

        elif isinstance(first_value, dict):
            if all(isinstance(v, dict) for v in values):
                result[key] = batch_aggregate(values)
            else:
                result[key] = values

        else:
            result[key] = values

    return result