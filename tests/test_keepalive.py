"""The server must not ping its clients to death.

`websockets` defaults to a 20 s keepalive ping with a 20 s timeout. A learn
step is CPU-bound Python: it holds the GIL, the event loop does not run, the
pong is not read, and the server closes a perfectly healthy connection with
1011. That is bad on its own, and worse than it looks: the per-environment
maps that carry the previous observation live inside the connection handler,
so the reconnect starts with empty ones and the next feedback reaches the
algorithm with nothing behind it. No error, just a corrupt transition.

The protocol already has its own liveness check - FEEDBACK_WAIT_TIMEOUT - so
the ping buys nothing. These tests pin that down, because the setting is one
keyword away from coming back by accident.
"""

import asyncio

import pytest

from plugrl_server.server import websocket_agent_server as mod
from plugrl_server.server.websocket_agent_server import WebSocketAgentServer


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


@pytest.fixture
def server():
    return WebSocketAgentServer(
        _Algorithm(),
        _CheckpointManager(),
        _MetricSink(),
        show_metric_table=False,
        show_progress_bar=False,
    )


def _serve_kwargs(server, monkeypatch):
    """Start the server against a stub `serve` and return how it was called."""
    seen = {}

    async def fake_serve(*args, **kwargs):
        seen.update(kwargs)
        server._lifecycle.stop_event.set()  # let run() fall straight through
        return None

    monkeypatch.setattr(mod._server, "serve", fake_serve)
    monkeypatch.setattr(server, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(server, "_scheduler_loop", lambda: asyncio.sleep(0))

    asyncio.run(server.run())
    return seen


def test_keepalive_pings_are_off(server, monkeypatch):
    """A learn step can outlast any ping interval, so there is no safe one."""
    assert _serve_kwargs(server, monkeypatch)["ping_interval"] is None


def test_the_frame_size_cap_is_still_off(server, monkeypatch):
    """An observation is megabytes; the 1 MiB default would reject it."""
    assert _serve_kwargs(server, monkeypatch)["max_size"] is None
