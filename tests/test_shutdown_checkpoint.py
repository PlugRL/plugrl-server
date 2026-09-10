"""A run that is stopped early must still be resumable.

Periodic saving is the algorithm's decision, and the gap can be enormous:
FPO saves once per ten learn cycles, which at its default buffer size is one
save per 9.8 million environment steps - several hours of a CPU-only run. A
run interrupted before the first of those used to lose everything, and
interrupting a run is the normal case.

So the server writes a checkpoint on the way out. These tests cover the three
things that has to get right: it happens, it does not happen after a fatal
error, and it cannot hang the shutdown.
"""

import asyncio

import pytest

from plugrl_server.server import websocket_agent_server as mod
from plugrl_server.server.websocket_agent_server import WebSocketAgentServer


class _Algorithm:
    """Only what the server constructor and the save path touch."""

    policy = None

    def get_total_training_steps(self):
        return 100

    def create_checkpoint(self):
        return {"weights": "pretend"}


class _CheckpointManager:
    def __init__(self):
        self.saved = []

    def save_checkpoint(self, checkpoint):
        self.saved.append(checkpoint)


class _MetricSink:
    def log_scalars(self, *_a, **_k):
        pass


@pytest.fixture
def server():
    manager = _CheckpointManager()
    s = WebSocketAgentServer(
        _Algorithm(),
        manager,
        _MetricSink(),
        show_metric_table=False,
        show_progress_bar=False,
    )
    return s, manager


def test_a_checkpoint_is_written_on_the_way_out(server):
    s, manager = server

    asyncio.run(s._save_on_exit())

    assert len(manager.saved) == 1, "an interrupted run has to leave something behind"


def test_nothing_is_written_after_a_fatal_error(server):
    """The state that produced a crash is not state worth resuming from."""
    s, manager = server
    s._lifecycle.fatal_reported = True

    asyncio.run(s._save_on_exit())

    assert manager.saved == []


def test_nothing_is_written_when_the_algorithm_finished_on_its_own(server):
    """The completion path already saved at this step; twice is waste.

    `_run_control_cycle` writes a checkpoint and only then asks for shutdown
    with SERVER_STOP_REASON, so that reason is an exact signal that a save
    has just happened.
    """
    from plugrl_protocol.websocket_protocol import SERVER_STOP_REASON

    s, manager = server
    s._lifecycle.close_reason = SERVER_STOP_REASON

    asyncio.run(s._save_on_exit())

    assert manager.saved == []


def test_an_interrupted_run_still_saves_even_though_it_closed_connections(server):
    """Any other close reason means nobody has saved: this is the real case."""
    s, manager = server
    s._lifecycle.close_reason = ""

    asyncio.run(s._save_on_exit())

    assert len(manager.saved) == 1


def test_a_held_model_lock_does_not_hang_the_shutdown(server, monkeypatch):
    """A learn step is asked to stop, but may take a moment to notice.

    Waiting forever for it turns Ctrl-C into a hang, which is worse than
    losing the checkpoint.
    """
    s, manager = server
    monkeypatch.setattr(mod, "SHUTDOWN_SAVE_TIMEOUT", 0.05)

    async def scenario():
        await s._model_lock.acquire()  # never released, as a stuck learn would
        await s._save_on_exit()

    asyncio.run(scenario())

    assert manager.saved == [], "it should have given up rather than blocked"


def test_a_failing_save_is_survivable(server, monkeypatch):
    """A full disk at shutdown must not turn into an unhandled exception."""
    s, _ = server

    async def explode():
        raise OSError("No space left on device")

    monkeypatch.setattr(s._training, "process_save", explode)

    asyncio.run(s._save_on_exit())  # must not raise


def test_shutdown_saves_before_it_tears_anything_down(server, monkeypatch):
    """Ordering matters: the checkpoint needs the model that is about to go."""
    s, _ = server
    order = []

    async def fake_save():
        order.append("save")

    async def fake_lifecycle_shutdown(**_kwargs):
        order.append("teardown")

    monkeypatch.setattr(s, "_save_on_exit", fake_save)
    monkeypatch.setattr(s._lifecycle, "shutdown", fake_lifecycle_shutdown)

    async def scenario():
        task = asyncio.create_task(asyncio.sleep(0))
        await s._shutdown(task, "test")

    asyncio.run(scenario())

    assert order == ["save", "teardown"]
