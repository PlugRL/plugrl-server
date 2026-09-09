"""A policy has to know how many environments it was asked about.

DummyPolicy inferred that by taking len() of the first value in the
observation dict. Observations arrive shaped {"images": {...}, "states":
{...}, "text": [...]}, so that measured the number of cameras. With no
cameras it measured zero, and the server returned an action array with a zero
dimension to a client that had asked for one action - no error, no warning,
just nothing. A proprioceptive robot would hit this on its first step.
"""

import numpy as np
import pytest

from plugrl_server.common.data_utils import numpy_tree_batch_size
from plugrl_server.policy.dummy_policy import DummyPolicy, DummyPolicyConfig


def _observation(batch: int, *, cameras: int = 2) -> dict:
    images = {
        f"cam{i}": np.zeros((batch, 8, 8, 3), dtype=np.uint8) for i in range(cameras)
    }
    return {
        "images": images,
        "states": {"robot_state": np.zeros((batch, 10), dtype=np.float32)},
        "text": np.asarray(["do something"] * batch),
    }


class TestNumpyTreeBatchSize:
    def test_reads_the_leading_dimension_of_a_leaf(self):
        assert numpy_tree_batch_size(_observation(4)) == 4

    def test_camera_count_does_not_change_the_answer(self):
        for cameras in (0, 1, 2, 5):
            assert numpy_tree_batch_size(_observation(3, cameras=cameras)) == 3

    def test_bare_array(self):
        assert numpy_tree_batch_size(np.zeros((7, 2))) == 7

    def test_raises_rather_than_guessing(self):
        with pytest.raises(TypeError):
            numpy_tree_batch_size({})
        with pytest.raises(TypeError):
            numpy_tree_batch_size({"images": {}, "states": {}})


class TestDummyPolicyBatchSize:
    @pytest.mark.parametrize("cameras", [0, 1, 2])
    @pytest.mark.parametrize("batch", [1, 4])
    def test_action_batch_matches_the_request(self, cameras, batch):
        policy = DummyPolicy(DummyPolicyConfig(discrete=False, action_dim=7))
        action, _ = policy.get_action_and_runtime_state(
            _observation(batch, cameras=cameras)
        )
        assert action.shape[0] == batch, (
            f"asked about {batch} envs with {cameras} cameras, "
            f"got an action of shape {action.shape}"
        )

    def test_no_camera_observation_still_produces_actions(self):
        """The case that silently returned nothing."""
        policy = DummyPolicy(DummyPolicyConfig(discrete=False, action_dim=7))
        action, _ = policy.get_action_and_runtime_state(_observation(2, cameras=0))
        assert 0 not in action.shape

    def test_discrete_actions_too(self):
        policy = DummyPolicy(DummyPolicyConfig(discrete=True, action_dim=4))
        action, _ = policy.get_action_and_runtime_state(_observation(3, cameras=0))
        assert action.shape[0] == 3

    def test_runtime_state_matches_the_action_batch(self):
        policy = DummyPolicy(DummyPolicyConfig(discrete=False, action_dim=7))
        action, state = policy.get_action_and_runtime_state(_observation(5, cameras=0))
        assert action.shape[0] == 5
        assert state["value"].shape[0] == 5
