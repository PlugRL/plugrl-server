"""The documentation's reference implementations have to keep working.

`examples/sac` is what docs/algorithm/custom_algorithm.md points a new
algorithm's author to, and `examples/lerobot` is the diffusion policy
docs/policy/dppo_policy.md points to. Nothing ran either of them, so a change
to the server could break them without a sound. One did: #104 made the server
discard, for every algorithm, frames an older policy collected and frames that
arrive while the algorithm wants to learn. That is right for PPO, FPO and DPPO.
SAC asks to learn after every frame it stores, so with several clients it
kept about one frame per round and discarded the rest. Algorithms now say
whether they are on-policy (`BaseAlgorithm.on_policy`), and SAC says it is
not.

The example itself had also stopped working, twice over. Its policy config
lacked `@dataclass`, so no field could be set. And it gave its replay buffer
a batched example action, so the first update died on a shape mismatch.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

from _wire import CheckpointManager, train
from plugrl_protocol.websocket_protocol import SERVER_STOP_REASON

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
CLIENTS = 3
OBS_DIM = 3


@pytest.fixture
def sac_modules():
    sys.path.insert(0, str(EXAMPLES / "sac"))
    try:
        import sac
        import sac_policy

        yield sac, sac_policy
    finally:
        sys.path.remove(str(EXAMPLES / "sac"))


def test_sac_is_given_every_frame_and_learns(sac_modules):
    sac, sac_policy = sac_modules
    torch.manual_seed(0)
    policy = sac_policy.SACPolicy(
        sac_policy.SACPolicyConfig(state_dim=OBS_DIM, action_dim=1, device="cpu")
    )
    algo = sac.SACAlgorithm(
        sac.SACAlgoConfig(
            global_steps=60,
            buffer_size=1000,
            batch_size=8,
            learning_starts=12,
            save_interval=10**9,
        ),
        policy,
    )
    learns = []
    learn_impl = algo.learn_impl

    def counted_learn_impl():
        learns.append(algo.global_step)
        return learn_impl()

    algo.learn_impl = counted_learn_impl

    server, reasons = train(
        algo, CheckpointManager(), clients=CLIENTS, obs_dim=OBS_DIM, episode=7
    )

    assert reasons == [SERVER_STOP_REASON] * CLIENTS
    assert server._discarded_frames == 0
    assert algo.global_step >= 60
    assert len(algo.replay_buffer) == algo.global_step
    # Past learning_starts it asks to learn after every frame, and the server
    # learns once per batch of feedback that arrives together - as CleanRL's
    # SAC takes one update per step of all its environments.
    assert learns and learns[0] >= 12
    assert len(learns) >= (algo.global_step - 12) // CLIENTS


def test_the_lerobot_example_imports():
    pytest.importorskip("lerobot")
    pytest.importorskip("diffusers")
    sys.path.insert(0, str(EXAMPLES / "lerobot"))
    try:
        import lerobot_diffusion

        assert lerobot_diffusion.UID == "lerobot-diffusion-policy"
    finally:
        sys.path.remove(str(EXAMPLES / "lerobot"))
