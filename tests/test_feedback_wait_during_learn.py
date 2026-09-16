"""A learn step must not look like a dead client.

The server sends an action and then waits `feedback_wait_timeout` for the
client's feedback. Silence past that means the client is gone - unless the
server is the one that is busy. E11 ran a learn step of 58 minutes behind this
60 s timeout: every iteration closed all ten connections, and each client lost
the feedback it was holding, because a client drops held feedback across a
reconnect (SPEC.md section 7.6).

So the wait is extended while a learn step is in flight, and a client is only
treated as gone when it is silent while the server is idle.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from plugrl_server.server.websocket_agent_server import WebSocketAgentServer

TIMEOUT = 0.05
PAYLOAD = b"packed-feedback"


class _Algorithm:
    policy = None

    def get_total_training_steps(self):
        return 100

    def create_checkpoint(self):
        return {"weights": "pretend"}


class _CheckpointManager:
    def save_checkpoint(self, checkpoint):
        pass


class _MetricSink:
    def log_scalars(self, *_a, **_k):
        pass


class _Socket:
    """A websocket that delivers its feedback after a given delay."""

    def __init__(self, deliver_after: float) -> None:
        self.remote_address = ("127.0.0.1", 12345)
        self._deliver_at = time.monotonic() + deliver_after
        self.closed_with: tuple | None = None
        self.recv_calls = 0

    async def recv(self) -> bytes:
        self.recv_calls += 1
        while True:
            remaining = self._deliver_at - time.monotonic()
            if remaining <= 0:
                return PAYLOAD
            await asyncio.sleep(min(remaining, TIMEOUT / 10))

    async def close(self, code=None, reason=None) -> None:
        self.closed_with = (code, reason)


@pytest.fixture
def server():
    server = WebSocketAgentServer(
        _Algorithm(),
        _CheckpointManager(),
        _MetricSink(),
        show_metric_table=False,
        show_progress_bar=False,
        feedback_wait_timeout=TIMEOUT,
    )
    return server


def test_feedback_that_arrives_is_returned(server):
    socket = _Socket(deliver_after=0.0)

    assert asyncio.run(server._recv_feedback(socket)) == PAYLOAD
    assert socket.closed_with is None


def test_a_silent_client_is_dropped_when_the_server_is_idle(server):
    socket = _Socket(deliver_after=10.0)

    started_at = time.monotonic()
    assert asyncio.run(server._recv_feedback(socket)) is None
    waited = time.monotonic() - started_at

    assert socket.closed_with is not None, "the connection was left open"
    assert waited < TIMEOUT * 5, f"waited {waited:.3f}s for one timeout of {TIMEOUT}s"


def test_the_wait_is_extended_while_a_learn_step_is_in_flight(server):
    server._learning = True
    socket = _Socket(deliver_after=TIMEOUT * 4)

    started_at = time.monotonic()
    assert asyncio.run(server._recv_feedback(socket)) == PAYLOAD
    waited = time.monotonic() - started_at

    assert socket.closed_with is None, "a learn step closed a live connection"
    assert waited > TIMEOUT * 2, "the wait did not outlast a single timeout"


def test_a_client_silent_after_the_learn_step_is_still_dropped(server):
    """The extension lasts as long as the learn does, not longer."""
    server._learning = True
    socket = _Socket(deliver_after=10.0)

    async def drive():
        task = asyncio.ensure_future(server._recv_feedback(socket))
        await asyncio.sleep(TIMEOUT * 2)
        assert not task.done(), "the wait was not extended while learning"
        server._learning = False
        return await task

    assert asyncio.run(drive()) is None
    assert socket.closed_with is not None


def test_the_timeout_is_configurable(server):
    assert server._feedback_wait_timeout == TIMEOUT

    default = WebSocketAgentServer(
        _Algorithm(),
        _CheckpointManager(),
        _MetricSink(),
        show_metric_table=False,
        show_progress_bar=False,
    )
    assert default._feedback_wait_timeout == 60.0
