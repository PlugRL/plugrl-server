import torch
from typing import get_type_hints, Any, List, Dict, Union


class TensorContainer:
    _fields: List[str] = []

    def __init__(self, **kwargs: torch.Tensor):
        unknown_keys = set(kwargs.keys()) - set(self._fields)
        if unknown_keys:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {', '.join(unknown_keys)}"
            )

        for field in self._fields:
            if field not in kwargs:
                raise ValueError(f"Missing required field: {field}")
            setattr(self, field, kwargs[field])

    def __getitem__(self, i: Any) -> "TensorContainer":
        new_items = {}
        for field in self._fields:
            item = getattr(self, field)
            new_items[field] = item[i]

        return self.__class__(**new_items)

    def __getattr__(self, name: str) -> Any:
        try:
            _fields = super().__getattribute__("_fields")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no _fields defined."
            )

        if not _fields:
            raise AttributeError(
                f"'{type(self).__name__}' object has no _fields defined."
            )

        try:
            first_tensor = super().__getattribute__(_fields[0])
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if callable(getattr(first_tensor, name, None)):

            def wrapper(*args, **kwargs):
                new_items = {}
                for field in _fields:
                    item = super(TensorContainer, self).__getattribute__(field)
                    new_items[field] = getattr(item, name)(*args, **kwargs)
                return self.__class__(**new_items)

            return wrapper
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def tensor_container(cls):
    hints = get_type_hints(cls)
    tensor_fields = [
        name for name in hints.keys() if name != "_fields" and name != "self"
    ]

    cls._fields = tensor_fields

    field_names = cls._fields

    if field_names:
        field_args_str = ", ".join([f"{f}: Any" for f in field_names])
        signature = f"*, {field_args_str}"
        init_body_args = ", ".join([f"{f}={f}" for f in field_names])
    else:
        signature = "**kwargs"
        init_body_args = ""

    init_source = f"""
def __init__(self, {signature}):
    TensorContainer.__init__(self, {init_body_args})
"""

    exec_globals = dict(
        TensorContainer=TensorContainer,
        Any=Any,
        List=List,
        Dict=Dict,
        Union=Union,
        torch=torch,
        Tensor=torch.Tensor,
    )
    exec_locals = {}

    exec(init_source, exec_globals, exec_locals)

    setattr(cls, "__init__", exec_locals["__init__"])

    return cls
