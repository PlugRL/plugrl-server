"""The server must not cap the frame size, and must survive a reconnect.

An observation is megabytes, so `max_size` has to stay off; the `websockets`
default of 1 MiB would reject a two-camera frame outright.

The keepalive is deliberately left at the library default. An earlier version
of this file asserted `ping_interval is None`, on the theory that a CPU-bound
learn step holds the event loop past the 20 s ping timeout. That theory was
measured and is false - `LocalTrainingBackend` runs `learn` through
`asyncio.to_thread`, and five learns of about 180 s each produced no timeout
at all. See `experiments/e8-keepalive-hypothesis/`.

What is true, and what `test_a_missing_step_state_is_reported` covers, is the
consequence of a reconnect whatever caused it: the per-environment maps live
in the connection handler, so the new connection starts empty.
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


def test_the_frame_size_cap_is_off(server, monkeypatch):
    """An observation is megabytes; the 1 MiB default would reject it."""
    assert _serve_kwargs(server, monkeypatch)["max_size"] is None


def test_compression_is_off(server, monkeypatch):
    """Observations are already-compressed image bytes; deflate only costs."""
    assert _serve_kwargs(server, monkeypatch)["compression"] is None
