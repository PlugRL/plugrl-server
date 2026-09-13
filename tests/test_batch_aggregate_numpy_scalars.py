"""Stacking per-env observations has to batch numpy scalars too.

Pi0Policy runs openpi's input transforms one observation at a time, then stacks
the results with batch_aggregate. openpi's LiberoInputs sets each image_mask
entry to np.True_ or np.False_ - a numpy scalar, not an ndarray. batch_aggregate
only stacked ndarrays, so every mask came back as a Python list, and
numpy_state_to_torch_tree rejected it. The first infer against a real pi05_libero
policy took the server down with "Unsupported model observation type: <class
'list'>".
"""

import numpy as np
import pytest
import torch

from plugrl_server.common.data_utils import batch_aggregate, numpy_state_to_torch_tree


def _transformed(i: int) -> dict:
    """One observation shaped the way openpi's LiberoInputs leaves it."""
    return {
        "state": np.zeros((32,), dtype=np.float64),
        "image": {
            "base_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
        "tokenized_prompt": np.full((8,), i, dtype=np.int64),
        "tokenized_prompt_mask": np.ones((8,), dtype=bool),
    }


@pytest.mark.parametrize("batch", [1, 3])
def test_image_masks_gain_a_batch_dimension(batch):
    stacked = batch_aggregate([_transformed(i) for i in range(batch)])

    for camera, expected in (
        ("base_0_rgb", True),
        ("left_wrist_0_rgb", True),
        ("right_wrist_0_rgb", False),
    ):
        mask = stacked["image_mask"][camera]
        assert isinstance(mask, np.ndarray), f"{camera} came back as {type(mask)}"
        assert mask.shape == (batch,)
        assert mask.dtype == np.bool_
        assert (mask == expected).all()


def test_the_stacked_observation_converts_to_torch():
    """The call that raised on the first infer."""
    tree = numpy_state_to_torch_tree(
        batch_aggregate([_transformed(0), _transformed(1)])
    )

    assert tree["image_mask"]["base_0_rgb"].dtype == torch.bool
    assert tuple(tree["image_mask"]["right_wrist_0_rgb"].shape) == (2,)
    assert tuple(tree["tokenized_prompt"].shape) == (2, 8)


@pytest.mark.parametrize(
    "scalars, dtype",
    [
        ([np.float32(0.5), np.float32(1.5)], np.float32),
        ([np.int64(3), np.int64(4)], np.int64),
        ([np.str_("a"), np.str_("bc")], np.dtype("<U2")),
    ],
)
def test_other_numpy_scalars_stack_the_same_way(scalars, dtype):
    stacked = batch_aggregate([{"x": s} for s in scalars])["x"]
    assert isinstance(stacked, np.ndarray)
    assert stacked.shape == (len(scalars),)
    assert stacked.dtype == dtype


def test_concat_leaves_scalars_alone():
    """A scalar has no axis to concatenate along, so concat keeps the old list."""
    values = batch_aggregate(
        [{"x": np.True_}, {"x": np.False_}], aggregate_method="concat"
    )["x"]
    assert isinstance(values, list)


def test_python_scalars_are_still_collected_as_a_list():
    assert batch_aggregate([{"x": True}, {"x": False}])["x"] == [True, False]
