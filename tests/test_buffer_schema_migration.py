import pytest

from plugrl_server.buffer.schema_migration import migrate_buffer_payload


def test_migrate_buffer_payload_identity_for_current_rollout_version() -> None:
    payload = dict(
        buffer_kind="rollout",
        buffer_schema_version=1,
        storage_format="numpy_tree",
        obs=dict(),
    )

    migrated = migrate_buffer_payload(
        payload,
        expected_kind="rollout",
        target_version=1,
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
            target_version=1,
        )


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
            target_version=1,
        )
