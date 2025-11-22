from openpi import transforms as _transforms
import difflib

_TRANSFORMS_DICT: dict[str, list[_transforms.DataTransformFn]] = {
    "pi05_libero": [_transforms.RepackTransform(
        {
            "observation/image": "images/base",
            "observation/wrist_image": "images/wrist",
            "observation/state": "states/eef",
            "prompt": "text",
        }
    )],
    "pi05_tiny_libero": [_transforms.RepackTransform(
        {
            "observation/image": "images/base",
            "observation/wrist_image": "images/wrist",
            "observation/state": "states/eef",
            "prompt": "text",
        }
    )],
}

def get_transform(config_name: str) -> list[_transforms.DataTransformFn]:
    """Get a config by name."""
    if config_name not in _TRANSFORMS_DICT:
        closest = difflib.get_close_matches(config_name, _TRANSFORMS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Transform '{config_name}' not found.{closest_str}")

    return _TRANSFORMS_DICT[config_name]