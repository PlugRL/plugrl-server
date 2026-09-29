"""A malformed client loses its connection, not everyone's training run.

Any exception in a connection handler is treated as fatal and stops the
server. A feedback message whose `info` could not be split into one entry per
environment reached an `assert` there, so a single client sending
`{"task": "pick"}` for two environments - which SPEC.md 5.4 describes as a
gap - shut the server down for every other client too. The docs audit
reproduced it: the connection closed with 1011 and the process exited.

Such a message is now a protocol error. That connection is closed with the
resync reason, as for any other malformed message, and the server goes on.
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch
import websockets
import websockets.asyncio.client as ws_client

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import SERVER_RESYNC_REASON, SERVER_STOP_REASON
from plugrl_server.algorithm.ppo.ppo import PPOAlgorithm
from plugrl_server.algorithm.ppo.ppo_config import PPOAlgoConfig
from plugrl_server.policy.gaussian.gaussian_policy import (
    GaussianPolicy,
    GaussianPolicyConfig,
)
from plugrl_server.server.websocket_agent_server import WebSocketAgentServer

OBS_DIM = 3


class _CheckpointManager:
    def save_checkpoint(self, checkpoint) -> None:
        pass


class _MetricSink:
    def log_scalars(self, *_a, **_k) -> None:
        pass


def _obs(rng: np.random.Generator, n: int) -> dict:
    return dict(
        images={},
        states={"obs": rng.normal(size=(n, OBS_DIM)).astype(np.float32)},
        text=["x"] * n,
    )


def _feedback(rng, n: int, info: dict) -> dict:
    return dict(
        message_type="feedback",
        env_indices=np.arange(n),
        step_ids=np.zeros(n, dtype=np.int64),
        data=dict(
            obs=_obs(rng, n),
            rewards=np.ones(n, np.float32),
            terminated=np.zeros(n, np.bool_),
            truncated=np.zeros(n, np.bool_),
            info=info,
        ),
    )


async def _client(port: int, n: int, info: dict) -> str:
    """Infer and feed back for `n` envs until closed; the close reason."""
    rng = np.random.default_rng(n)
    packer = msgpack_numpy.Packer()
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
                            data=_obs(rng, n),
                            env_indices=np.arange(n),
                            step_ids=np.zeros(n, dtype=np.int64),
                        )
                    )
                )
                await ws.recv()  # the action
                await ws.send(packer.pack(_feedback(rng, n, info)))
        except websockets.ConnectionClosed as closed:
            return closed.rcvd.reason if closed.rcvd is not None else ""


def test_a_malformed_info_closes_its_connection_and_training_goes_on():
    torch.manual_seed(0)
    algo = PPOAlgorithm(
        PPOAlgoConfig(buffer_size=10, batch_size=5, update_epochs=1, train_itrs=2),
        GaussianPolicy(
            GaussianPolicyConfig(obs_dim=OBS_DIM, action_dim=1, device="cpu")
        ),
    )
    algo.init_optimizers()

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
            asyncio.gather(
                _client(port, 2, {"task": "pick"}),  # no length-2 array to split by
                _client(port, 1, {}),
            ),
            timeout=120,
        )
        await asyncio.wait_for(run, timeout=60)
        return server, reasons

    server, (bad, good) = asyncio.run(main())

    assert bad == SERVER_RESYNC_REASON
    assert good == SERVER_STOP_REASON
    assert not server._lifecycle.fatal_reported
    assert algo.curr_train_itrs == 2
