import pytest
import numpy as np

from plugrl_server.buffer.replay_buffer import ReplayBuffer
from plugrl_server.buffer.rollout_buffer import RolloutBuffer
from plugrl_server.buffer.schema_migration import migrate_buffer_payload


def test_migrate_buffer_payload_identity_for_current_rollout_version() -> None:
    payload = dict(
        buffer_kind="rollout",
        buffer_schema_version=2,
        storage_format="numpy_tree",
        train_state=dict(),
    )

    migrated = migrate_buffer_payload(
        payload,
        expected_kind="rollout",
        target_version=2,
    )

    assert migrated is payload


def test_migrate_buffer_payload_rejects_wrong_kind() -> None:
    payload = dict(
        buffer_kind="replay",
        buffer_schema_version=1,
        storage_format="numpy_tree",
    )

    with pytest.raises(ValueError, match="Unexpected buffer kind"):
        migrate_buffer_payload(
            payload,
            expected_kind="rollout",
            target_version=2,
        )


def test_migrate_rollout_payload_v1_to_v2_renames_fields() -> None:
    payload = dict(
        buffer_kind="rollout",
        buffer_schema_version=1,
        storage_format="numpy_tree",
        obs=dict(x=np.zeros((1, 2), dtype=np.float32)),
        obs_spec=dict(x=dict(shape=(2,), dtype="float32")),
    )

    migrated = migrate_buffer_payload(
        payload,
        expected_kind="rollout",
        target_version=2,
    )

    assert migrated["buffer_schema_version"] == 2
    assert "obs" not in migrated
    assert "obs_spec" not in migrated
    assert "train_state" in migrated
    assert "train_state_spec" in migrated


def test_migrate_replay_payload_v1_to_v2_renames_fields() -> None:
    payload = dict(
        buffer_kind="replay",
        buffer_schema_version=1,
        storage_format="numpy_tree",
        obs=dict(state=np.zeros((1, 3), dtype=np.float32)),
        next_obs=dict(state=np.ones((1, 3), dtype=np.float32)),
        obs_spec=dict(state=dict(shape=(3,), dtype="float32")),
    )

    migrated = migrate_buffer_payload(
        payload,
        expected_kind="replay",
        target_version=2,
    )

    assert migrated["buffer_schema_version"] == 2
    assert "obs" not in migrated
    assert "next_obs" not in migrated
    assert "obs_spec" not in migrated
    assert "state" in migrated
    assert "next_state" in migrated
    assert "state_spec" in migrated


def test_rollout_buffer_round_trip_restores_full_state() -> None:
    example_train_state = dict(
        obs=dict(
            x=np.zeros((1, 2, 3), dtype=np.float32),
            cond=dict(state=np.zeros((1, 4), dtype=np.float32)),
        ),
        action=np.zeros((1, 2), dtype=np.float32),
        logprob=np.zeros((1, 2), dtype=np.float32),
        value=np.zeros((1,), dtype=np.float32),
    )
    buffer = RolloutBuffer(buffer_size=4, example_train_state=example_train_state)

    train_state_0 = dict(
        obs=dict(
            x=np.arange(6, dtype=np.float32).reshape(1, 2, 3),
            cond=dict(state=np.array([[1, 2, 3, 4]], dtype=np.float32)),
        ),
        action=np.array([[0.1, 0.2]], dtype=np.float32),
        logprob=np.array([[0.3, 0.4]], dtype=np.float32),
        value=np.array([0.5], dtype=np.float32),
    )
    train_state_1 = dict(
        obs=dict(
            x=(np.arange(6, dtype=np.float32) + 10).reshape(1, 2, 3),
            cond=dict(state=np.array([[5, 6, 7, 8]], dtype=np.float32)),
        ),
        action=np.array([[1.1, 1.2]], dtype=np.float32),
        logprob=np.array([[1.3, 1.4]], dtype=np.float32),
        value=np.array([1.5], dtype=np.float32),
    )

    first_node = buffer.add_frame(
        prev_node=(-1, buffer.buffer_signature),
        train_state=train_state_0,
        reward=1.0,
        terminated=False,
        truncated=False,
        last_value=np.array([9.0], dtype=np.float32),
        next_terminated=False,
        next_truncated=False,
    )
    buffer.add_frame(
        prev_node=first_node,
        train_state=train_state_1,
        reward=2.0,
        terminated=True,
        truncated=False,
        last_value=np.array([10.0], dtype=np.float32),
        next_terminated=True,
        next_truncated=False,
    )
    buffer.advantages[:2] = np.array([11.0, 12.0], dtype=np.float32)
    buffer.returns[:2] = np.array([13.0, 14.0], dtype=np.float32)
    buffer.finish_rollout(dict(reward=123.0, length=7))

    payload = buffer.as_dict()
    restored = RolloutBuffer(buffer_size=4, example_train_state=example_train_state)
    restored.load_dict(payload)

    assert restored.idx == buffer.idx
    assert restored.buffer_signature == buffer.buffer_signature
    assert restored.episode_info_buffer == buffer.episode_info_buffer
    _assert_numpy_tree_equal(
        restored.train_state_storage.get_item(slice(None, restored.idx)),
        buffer.train_state_storage.get_item(slice(None, buffer.idx)),
    )
    np.testing.assert_array_equal(
        restored.actions[: restored.idx], buffer.actions[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.logprobs[: restored.idx], buffer.logprobs[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.rewards[: restored.idx], buffer.rewards[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.values[: restored.idx], buffer.values[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.last_values[: restored.idx], buffer.last_values[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.advantages[: restored.idx], buffer.advantages[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.returns[: restored.idx], buffer.returns[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.dones[: restored.idx], buffer.dones[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.next_done[: restored.idx], buffer.next_done[: buffer.idx]
    )
    np.testing.assert_array_equal(
        restored.next_indices[: restored.idx], buffer.next_indices[: buffer.idx]
    )


def test_replay_buffer_round_trip_restores_full_state() -> None:
    example_state = dict(
        state=np.zeros((1, 3), dtype=np.float32),
        aux=dict(flag=np.zeros((1, 2), dtype=np.float32)),
    )
    example_action = np.zeros((2,), dtype=np.float32)
    buffer = ReplayBuffer(
        buffer_size=4, example_state=example_state, example_action=example_action
    )

    buffer.add_frame(
        prev_node=(-1, buffer.buffer_signature),
        state=dict(
            state=np.array([1, 2, 3], dtype=np.float32),
            aux=dict(flag=np.array([4, 5], dtype=np.float32)),
        ),
        next_state=dict(
            state=np.array([6, 7, 8], dtype=np.float32),
            aux=dict(flag=np.array([9, 10], dtype=np.float32)),
        ),
        action=np.array([0.1, 0.2], dtype=np.float32),
        reward=1.5,
        done=False,
        timeout=True,
    )
    buffer.add_frame(
        prev_node=(-1, buffer.buffer_signature),
        state=dict(
            state=np.array([11, 12, 13], dtype=np.float32),
            aux=dict(flag=np.array([14, 15], dtype=np.float32)),
        ),
        next_state=dict(
            state=np.array([16, 17, 18], dtype=np.float32),
            aux=dict(flag=np.array([19, 20], dtype=np.float32)),
        ),
        action=np.array([1.1, 1.2], dtype=np.float32),
        reward=2.5,
        done=True,
        timeout=False,
    )

    payload = buffer.as_dict()
    restored = ReplayBuffer(
        buffer_size=4, example_state=example_state, example_action=example_action
    )
    restored.load_dict(payload)

    assert restored.idx == buffer.idx
    assert restored.full() == buffer.full()
    assert restored.buffer_signature == buffer.buffer_signature
    upper = len(buffer)
    _assert_numpy_tree_equal(
        restored.state_storage.get_item(slice(None, upper)),
        buffer.state_storage.get_item(slice(None, upper)),
    )
    _assert_numpy_tree_equal(
        restored.next_state_storage.get_item(slice(None, upper)),
        buffer.next_state_storage.get_item(slice(None, upper)),
    )
    np.testing.assert_array_equal(restored.actions[:upper], buffer.actions[:upper])
    np.testing.assert_array_equal(restored.rewards[:upper], buffer.rewards[:upper])
    np.testing.assert_array_equal(restored.dones[:upper], buffer.dones[:upper])
    np.testing.assert_array_equal(restored.timeouts[:upper], buffer.timeouts[:upper])


def _assert_numpy_tree_equal(left: object, right: object) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
        return
    assert isinstance(left, dict)
    assert isinstance(right, dict)
    assert left.keys() == right.keys()
    for key in left:
        _assert_numpy_tree_equal(left[key], right[key])


def test_migrate_buffer_payload_rejects_unregistered_version_jump() -> None:
    payload = dict(
        buffer_kind="rollout",
        buffer_schema_version=0,
        storage_format="numpy_tree",
    )

    with pytest.raises(ValueError, match="No migration registered"):
        migrate_buffer_payload(
            payload,
            expected_kind="rollout",
            target_version=2,
        )
