from __future__ import annotations

from collections.abc import Callable
from typing import Literal

BufferKind = Literal["rollout", "replay"]
MigrationFn = Callable[[dict], dict]


def _migrate_rollout_v1_to_v2(payload: dict) -> dict:
    migrated = payload.copy()
    migrated["buffer_schema_version"] = 2
    migrated["train_state"] = migrated.pop("obs")
    migrated["train_state_spec"] = migrated.pop("obs_spec")
    return migrated


def _migrate_replay_v1_to_v2(payload: dict) -> dict:
    migrated = payload.copy()
    migrated["buffer_schema_version"] = 2
    migrated["state"] = migrated.pop("obs")
    migrated["next_state"] = migrated.pop("next_obs")
    migrated["state_spec"] = migrated.pop("obs_spec")
    return migrated


def _migrate_rollout_v2_to_v3(payload: dict) -> dict:
    migrated = payload.copy()
    dones = migrated["dones"]
    next_done = migrated["next_done"]
    migrated["buffer_schema_version"] = 3
    migrated["terminated"] = dones.copy()
    migrated["truncated"] = dones.copy()
    migrated["truncated"].fill(False)
    migrated["next_terminated"] = next_done.copy()
    migrated["next_truncated"] = next_done.copy()
    migrated["next_truncated"].fill(False)
    return migrated


BUFFER_MIGRATORS: dict[tuple[BufferKind, int, int], MigrationFn] = dict(
    [
        (("rollout", 1, 2), _migrate_rollout_v1_to_v2),
        (("rollout", 2, 3), _migrate_rollout_v2_to_v3),
        (("replay", 1, 2), _migrate_replay_v1_to_v2),
    ]
)


def migrate_buffer_payload(
    payload: dict,
    *,
    expected_kind: BufferKind,
    target_version: int,
) -> dict:
    buffer_kind = payload.get("buffer_kind")
    if buffer_kind is None:
        raise ValueError("Buffer payload is missing 'buffer_kind'.")
    if buffer_kind != expected_kind:
        raise ValueError(
            f"Unexpected buffer kind {buffer_kind!r}, expected {expected_kind!r}."
        )

    current_version = payload.get("buffer_schema_version")
    if current_version is None:
        raise ValueError("Buffer payload is missing 'buffer_schema_version'.")

    migrated = payload
    while current_version != target_version:
        step = current_version + 1
        migrator = BUFFER_MIGRATORS.get((expected_kind, current_version, step))
        if migrator is None:
            raise ValueError(
                f"No migration registered for {buffer_kind!r} buffer schema "
                f"{current_version} -> {target_version}."
            )
        migrated = migrator(migrated)
        current_version = migrated.get("buffer_schema_version")
        if current_version is None:
            raise ValueError(
                "Migrated buffer payload is missing 'buffer_schema_version'."
            )

    return migrated
