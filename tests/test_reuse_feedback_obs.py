"""SPEC.md section 10.1: a client may leave out the observations the server has.

Every observation used to cross the link twice: in the feedback that ends a
chunk, and again in the next infer. With `reuse-feedback-obs` the infer
marks the rows the server already holds and sends only the others. What the
policy and the algorithm see must not change, so the test that matters is
that a client reusing observations trains the same weights, bit for bit, as
one sending them all. The rest checks the edges: the server lists the
feature, and a reuse it cannot honour closes that connection only.
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch
import websockets
import websockets.asyncio.client as ws_client

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.reuse import REUSE_FEEDBACK_OBS
from plugrl_protocol.websocket_protocol import SERVER_RESYNC_REASON, SERVER_STOP_REASON
from plugrl_server.algorithm.ppo.ppo import PPOAlgorithm
from plugrl_server.algorithm.ppo.ppo_config import PPOAlgoConfig
from plugrl_server.policy.gaussian.gaussian_policy import (
    GaussianPolicy,
    GaussianPolicyConfig,
)
from plugrl_server.server.websocket_agent_server import WebSocketAgentServer

OBS_DIM = 3
ENVS = 2
EPISODES = (3, 4)  # per env, so a reset row and a reused row share an infer


class _CheckpointManager:
    def save_checkpoint(self, checkpoint) -> None:
        pass


class _MetricSink:
    def log_scalars(self, *_a, **_k) -> None:
        pass


def _rows(rows: list[np.ndarray]) -> dict:
    """An observation batched from per-env rows of `OBS_DIM` values."""
    states = np.stack(rows) if rows else np.zeros((0, OBS_DIM), np.float32)
    return dict(images={}, states={"obs": states}, text=["x"] * len(rows))


def _algo() -> PPOAlgorithm:
    torch.manual_seed(0)
    np.random.seed(0)
    algo = PPOAlgorithm(
        PPOAlgoConfig(buffer_size=12, batch_size=6, update_epochs=2, train_itrs=2),
        GaussianPolicy(
            GaussianPolicyConfig(obs_dim=OBS_DIM, action_dim=1, device="cpu")
        ),
    )
    algo.init_optimizers()
    return algo


async def _client(port: int, *, reuse: bool, counts: dict) -> str:
    """`ENVS` envs on one connection until closed; the close reason.

    Both kinds of client draw the same observations in the same order. The
    one with `reuse` sends an env's row only after a reset, and otherwise
    marks it as the observation its last feedback carried.
    """
    rng = np.random.default_rng(7)
    packer = msgpack_numpy.Packer()
    current = [rng.normal(size=OBS_DIM).astype(np.float32) for _ in range(ENVS)]
    steps = [0] * ENVS
    held = [False] * ENVS  # whether the server holds this env's current row
    async with ws_client.connect(
        f"ws://127.0.0.1:{port}", max_size=None, compression=None
    ) as ws:
        metadata = msgpack_numpy.unpackb(await ws.recv())["data"]
        counts["features"] = metadata.get("features")
        try:
            while True:
                infer = dict(
                    message_type="infer",
                    env_indices=np.arange(ENVS),
                    step_ids=np.asarray(steps, dtype=np.int64),
                )
                if reuse:
                    infer["reuse"] = np.asarray(held, dtype=np.bool_)
                    infer["data"] = _rows(
                        [current[i] for i in range(ENVS) if not held[i]]
                    )
                    counts["reused"] += sum(held)
                    counts["sent"] += ENVS - sum(held)
                else:
                    infer["data"] = _rows(current)
                await ws.send(packer.pack(infer))
                await ws.recv()  # the action

                ended = [steps[i] + 1 == EPISODES[i] for i in range(ENVS)]
                following = [
                    rng.normal(size=OBS_DIM).astype(np.float32) for _ in range(ENVS)
                ]
                info = {}
                if any(ended):
                    info = dict(
                        episode=dict(
                            r=np.asarray([float(EPISODES[i]) for i in range(ENVS)]),
                            l=np.asarray(EPISODES),
                            s=np.zeros(ENVS),
                            mask=np.asarray(ended),
                        )
                    )
                await ws.send(
                    packer.pack(
                        dict(
                            message_type="feedback",
                            env_indices=np.arange(ENVS),
                            step_ids=np.asarray(steps, dtype=np.int64),
                            data=dict(
                                obs=_rows(following),
                                rewards=np.ones(ENVS, np.float32),
                                terminated=np.zeros(ENVS, np.bool_),
                                truncated=np.asarray(ended),
                                info=info,
                            ),
                        )
                    )
                )
                for i in range(ENVS):
                    if ended[i]:
                        # The feedback carried the terminal observation; the
                        # next chunk starts from a reset the server has not seen.
                        current[i] = rng.normal(size=OBS_DIM).astype(np.float32)
                        steps[i], held[i] = 0, False
                    else:
                        current[i] = following[i]
                        steps[i], held[i] = steps[i] + 1, True
        except websockets.ConnectionClosed as closed:
            return closed.rcvd.reason if closed.rcvd is not None else ""


async def _bad_reuse(port: int) -> str:
    """Reuse env 0's observation on a connection that never fed it back."""
    async with ws_client.connect(
        f"ws://127.0.0.1:{port}", max_size=None, compression=None
    ) as ws:
        await ws.recv()  # metadata
        await ws.send(
            msgpack_numpy.Packer().pack(
                dict(
                    message_type="infer",
                    data=_rows([]),
                    env_indices=np.asarray([0]),
                    step_ids=np.asarray([0]),
                    reuse=np.asarray([True]),
                )
            )
        )
        try:
            await ws.recv()
        except websockets.ConnectionClosed as closed:
            return closed.rcvd.reason if closed.rcvd is not None else ""
        return "answered"


def _train(*clients) -> tuple[WebSocketAgentServer, PPOAlgorithm, list[str]]:
    algo = _algo()

    async def main():
        server = WebSocketAgentServer(
            algo,
            _CheckpointManager(),
            _MetricSink(),
            host="127.0.0.1",
            port=0,
            show_metric_table=False,
            show_progress_bar=False,
        )
        server._install_signal_handlers = lambda: None
        run = asyncio.create_task(server.run())
        while server._server is None:
            await asyncio.sleep(0.01)
        port = server._server.sockets[0].getsockname()[1]
        reasons = await asyncio.wait_for(
            asyncio.gather(*(c(port) for c in clients)), timeout=120
        )
        await asyncio.wait_for(run, timeout=60)
        return server, list(reasons)

    server, reasons = asyncio.run(main())
    return server, algo, reasons


def test_a_client_that_reuses_observations_trains_the_same_weights():
    full, reused = dict(reused=0, sent=0), dict(reused=0, sent=0)

    server_full, algo_full, reasons_full = _train(
        lambda port: _client(port, reuse=False, counts=full)
    )
    server_reused, algo_reused, reasons_reused = _train(
        lambda port: _client(port, reuse=True, counts=reused)
    )

    assert reasons_full == reasons_reused == [SERVER_STOP_REASON]
    assert reused["features"] == [REUSE_FEEDBACK_OBS]
    # It did reuse, and it did send the rows after each reset.
    assert reused["reused"] > reused["sent"] > ENVS
    # The server's count is what an operator sees. The client's last infer
    # may meet the stop instead of being read, so it can be one infer short.
    counted = server_reused._runtime_metrics()["server"]["reused_observations"]
    assert reused["reused"] - ENVS <= counted <= reused["reused"]
    assert server_full._runtime_metrics()["server"]["reused_observations"] == 0
    assert algo_full.curr_train_itrs == algo_reused.curr_train_itrs == 2
    weights_full = algo_full.policy.state_dict()
    weights_reused = algo_reused.policy.state_dict()
    assert weights_full.keys() == weights_reused.keys()
    for name in weights_full:
        assert torch.equal(weights_full[name], weights_reused[name]), name


def test_a_reuse_it_cannot_honour_closes_that_connection_only():
    counts = dict(reused=0, sent=0)

    server, algo, (bad, good) = _train(
        _bad_reuse, lambda port: _client(port, reuse=True, counts=counts)
    )

    assert bad == SERVER_RESYNC_REASON
    assert good == SERVER_STOP_REASON
    assert not server._lifecycle.fatal_reported
    assert algo.curr_train_itrs == 2
