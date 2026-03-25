from dataclasses import dataclass
from typing import Any

from plugrl_protocol.websocket_protocol import MessageType


class ProtocolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MetadataMessage:
    data: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "message_type": str(MessageType.METADATA),
            "data": self.data,
        }


@dataclass(frozen=True)
class ActionMessage:
    data: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "message_type": str(MessageType.ACTION),
            "data": self.data,
        }


@dataclass(frozen=True)
class InferRequestMessage:
    data: Any
    env_indices: Any
    step_ids: Any


@dataclass(frozen=True)
class FeedbackData:
    obs: Any
    rewards: Any
    terminated: Any
    truncated: Any
    info: Any


@dataclass(frozen=True)
class FeedbackRequestMessage:
    env_indices: Any
    step_ids: Any
    data: FeedbackData


def parse_infer_request(payload: dict[str, Any]) -> InferRequestMessage:
    _expect_message_type(payload, MessageType.INFER)
    return InferRequestMessage(
        data=payload["data"],
        env_indices=payload["env_indices"],
        step_ids=payload["step_ids"],
    )


def parse_feedback_request(payload: dict[str, Any]) -> FeedbackRequestMessage:
    _expect_message_type(payload, MessageType.FEEDBACK)
    data = payload["data"]
    return FeedbackRequestMessage(
        env_indices=payload["env_indices"],
        step_ids=payload["step_ids"],
        data=FeedbackData(
            obs=data["obs"],
            rewards=data["rewards"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            info=data["info"],
        ),
    )


def _expect_message_type(payload: dict[str, Any], expected: MessageType) -> None:
    actual = payload.get("message_type")
    if actual != str(expected):
        raise ProtocolValidationError(
            f"Expected message_type={str(expected)!r}, received {actual!r}"
        )
