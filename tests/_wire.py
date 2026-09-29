"""Env clients that speak the wire protocol to a real server, in-process.

Each client runs one env whose observation is `states["obs"]` of `obs_dim`
random values, and answers every action at once with reward 1, cutting the
episode every `episode` steps. Used by tests that need the server's own
scheduling, locking and learning, not a stand-in for them.
"""

from __future__ import annotations

import asyncio

import numpy as np
import websockets
import websockets.asyncio.client as ws_client

from plugrl_protocol import msgpack_numpy
from plugrl_server.server.websocket_agent_server import WebSocketAgentServer


class CheckpointManager:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def save_checkpoint(self, checkpoint) -> None:
        self.steps.append(checkpoint.step)


class MetricSink:
    def log_scalars(self, *_a, **_k) -> None:
        pass


def _obs(rng: np.random.Generator, obs_dim: int) -> dict:
    return dict(
        images={},
        states={"obs": rng.normal(size=(1, obs_dim)).astype(np.float32)},
        text=["x"],
    )


async def client(port: int, seed: int, *, obs_dim: int, episode: int) -> str:
    """Infer, step, feed back, as fast as the server answers; one env.

    Returns the reason the server gave for closing the connection.
    """
    rng = np.random.default_rng(seed)
    packer = msgpack_numpy.Packer()
    step_id = 0
    obs = _obs(rng, obs_dim)
    async with ws_client.connect(
        f"ws://127.0.0.1:{port}", max_size=None, compression=None
    ) as ws:
        await ws.recv()  # metadata
        try:
            while True:
                await ws.send(
                    packer.pack(
                        dict(
                            message_type="infer",
                            data=obs,
                            env_indices=np.array([0]),
                            step_ids=np.array([step_id]),
                        )
                    )
                )
                await ws.recv()  # the action
                obs = _obs(rng, obs_dim)
                truncated = step_id + 1 == episode
                info = {}
                if truncated:
                    info = dict(
                        episode=dict(
                            r=np.array([float(episode)]),
                            l=np.array([episode]),
                            s=np.array([0.0]),
                            mask=np.array([True]),
                        )
                    )
                await ws.send(
                    packer.pack(
                        dict(
                            message_type="feedback",
                            env_indices=np.array([0]),
                            step_ids=np.array([step_id]),
                            data=dict(
                                obs=obs,
                                rewards=np.array([1.0], np.float32),
                                terminated=np.array([False]),
                                truncated=np.array([truncated]),
                                info=info,
                            ),
                        )
                    )
                )
                step_id = 0 if truncated else step_id + 1
        except websockets.ConnectionClosed as closed:
            return closed.rcvd.reason if closed.rcvd is not None else ""


def train(
    algo, manager: CheckpointManager, *, clients: int, obs_dim: int, episode: int
) -> tuple[WebSocketAgentServer, list[str]]:
    """Serve `algo` to `clients` clients until it stops; the server and close reasons."""

    async def main() -> tuple[WebSocketAgentServer, list[str]]:
        server = WebSocketAgentServer(
            algo,
            manager,
            MetricSink(),
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
            asyncio.gather(
                *(
                    client(port, seed, obs_dim=obs_dim, episode=episode)
                    for seed in range(clients)
                )
            ),
            timeout=120,
        )
        await asyncio.wait_for(run, timeout=60)
        return server, list(reasons)

    return asyncio.run(main())
