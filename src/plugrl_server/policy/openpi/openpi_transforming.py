import difflib

from openpi import transforms as _transforms

_TRANSFORMS_DICT: dict[str, list[_transforms.DataTransformFn]] = {
    "pi05_libero": [
        _transforms.RepackTransform(
            {
                "observation/image": "images/base",
                "observation/wrist_image": "images/wrist",
                "observation/state": "states/eef",
                "prompt": "text",
            }
        )
    ],
    "pi05_tiny_libero": [
        _transforms.RepackTransform(
            {
                "observation/image": "images/base",
                "observation/wrist_image": "images/wrist",
                "observation/state": "states/eef",
                "prompt": "text",
            }
        )
    ],
}

_ROBOCASA_TRANSFORMS: list[_transforms.DataTransformFn] = [
    _transforms.RepackTransform(
        {
            "images": {
                "robot0_agentview_left": "images/robot0_agentview_left",
                "robot0_agentview_right": "images/robot0_agentview_right",
                "robot0_eye_in_hand": "images/robot0_eye_in_hand",
            },
            "state": "states/state",
            "prompt": "text",
        }
    )
]


def get_transform(config_name: str) -> list[_transforms.DataTransformFn]:
    """Get a config by name."""
    if "robocasa" in config_name.lower():
        return _ROBOCASA_TRANSFORMS

    if config_name not in _TRANSFORMS_DICT:
        closest = difflib.get_close_matches(
            config_name, _TRANSFORMS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Transform '{config_name}' not found.{closest_str}")

    return _TRANSFORMS_DICT[config_name]
