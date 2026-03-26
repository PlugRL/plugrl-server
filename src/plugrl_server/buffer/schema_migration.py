from __future__ import annotations

from typing import Literal

BufferKind = Literal["rollout", "replay"]


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

    if current_version == target_version:
        return payload

    raise ValueError(
        f"No migration registered for {buffer_kind!r} buffer schema "
        f"{current_version} -> {target_version}."
    )
