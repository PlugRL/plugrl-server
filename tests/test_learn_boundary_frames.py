"""A buffer trains only on frames that the policy learning from it collected.

The server infers for every connected client in one batch, so the round in
which a buffer fills usually leaves frames over: actions already sent, whose
feedback is still on its way. Which task took the model lock first decided
what happened to them:

- feedback that arrived before the learn step was refused by the full buffer
  but still counted in `global_step`;
- feedback that arrived during the learn step waited, then went into the next
  buffer carrying the log-probability and value of the policy before the
  update.

The scheduler also inferred before it checked whether to learn. Once a buffer
was full, a whole further round could be served by the old policy and land in
the next buffer the same way.

Measured with PPO on HalfCheetah and a buffer of 256:
- With three clients, each buffer held zero to three such frames, and four
  frames were counted but never stored. Checkpoints were named 258, 516, 774
  and so on, instead of 256, 512 and 768.
- With four clients, every buffer began with four.

This runs PPO behind the real server, with three clients that answer at once
and a buffer that three does not divide. It checks every buffer at the moment
it is about to be learned from.
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch
import websockets
import websockets.asyncio.client as ws_client

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import SERVER_STOP_REASON
from plugrl_server.algorithm.ppo.ppo import PPOAlgorithm
from plugrl_server.algorithm.ppo.ppo_config import PPOAlgoConfig
from plugrl_server.algorithm.train_utils import move_batch_to_device
from plugrl_server.policy.gaussian.gaussian_policy import (
    GaussianPolicy,
    GaussianPolicyConfig,
)
from plugrl_server.server.websocket_agent_server import WebSocketAgentServer

OBS_DIM = 3
BUFFER = 10
ITERS = 4
CLIENTS = 3
EPISODE = 7


class _CheckpointManager:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def save_checkpoint(self, checkpoint) -> None:
        self.steps.append(checkpoint.step)


class _MetricSink:
    def log_scalars(self, *_a, **_k) -> None:
        pass


def _algo() -> PPOAlgorithm:
    torch.manual_seed(0)
    policy = GaussianPolicy(
        GaussianPolicyConfig(obs_dim=OBS_DIM, action_dim=1, device="cpu")
    )
    algo = PPOAlgorithm(
        PPOAlgoConfig(
            buffer_size=BUFFER,
            batch_size=5,
            update_epochs=2,
            train_itrs=ITERS,
            save_interval=1,
        ),
        policy,
    )
    algo.init_optimizers()
    return algo


def _frames_from_another_policy(algo: PPOAlgorithm) -> int:
    """Buffered frames whose stored log-probability this policy does not give."""
    buf = algo.rollout_buffer
    n = len(buf)
    batch = buf.collate_fn([buf[i] for i in range(n)])
    obs, action, stored = move_batch_to_device(batch, device=algo.policy.device)[:3]
    with torch.no_grad():
        now, _, _ = algo.policy.evaluate_actions(obs, action)
    gap = (now.reshape(n, -1) - stored.reshape(n, -1)).abs().amax(dim=1)
    return int((gap > 1e-4).sum())


def _obs(rng: np.random.Generator) -> dict:
    return dict(
        images={},
        states={"obs": rng.normal(size=(1, OBS_DIM)).astype(np.float32)},
        text=["x"],
    )


async def _client(port: int, seed: int) -> str:
    """Infer, step, feed back, as fast as the server answers; one env."""
    rng = np.random.default_rng(seed)
    packer = msgpack_numpy.Packer()
    step_id = 0
    obs = _obs(rng)
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
                obs = _obs(rng)
                truncated = step_id + 1 == EPISODE
                info = {}
                if truncated:
                    info = dict(
                        episode=dict(
                            r=np.array([float(EPISODE)]),
                            l=np.array([EPISODE]),
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


def _train(algo: PPOAlgorithm, manager: _CheckpointManager) -> list[str]:
    async def main() -> list[str]:
        server = WebSocketAgentServer(
            algo,
            manager,
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
            asyncio.gather(*(_client(port, seed) for seed in range(CLIENTS))),
            timeout=120,
        )
        await asyncio.wait_for(run, timeout=60)
        return list(reasons)

    return asyncio.run(main())


def test_every_buffer_is_its_own_policys_and_checkpoints_count_whole_buffers():
    algo = _algo()
    seen: list[tuple[int, int]] = []
    pre_learn = algo.pre_learn

    def checked_pre_learn() -> None:
        seen.append((len(algo.rollout_buffer), _frames_from_another_policy(algo)))
        pre_learn()

    algo.pre_learn = checked_pre_learn
    manager = _CheckpointManager()

    reasons = _train(algo, manager)

    assert reasons == [SERVER_STOP_REASON] * CLIENTS
    assert seen == [(BUFFER, 0)] * ITERS
    assert manager.steps == [BUFFER * (k + 1) for k in range(ITERS)]
    assert algo.global_step == BUFFER * ITERS
