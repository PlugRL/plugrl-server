"""DPPO has to keep a normalising policy's observation statistics up to date.

`fpo-policy` normalises its input with running statistics stored as buffers,
and does not maintain them itself. `FPOAlgorithm.pre_learn` did, behind
`isinstance(self.policy, FPOPolicy)`; nothing else did. So when E17 drove the
same policy with DPPO, `obs_stats_count` stayed at 0.0 for a hundred
iterations - mean zero, standard deviation one, the identity - on
HalfCheetah, whose observation dimensions range in standard deviation from
0.17 to 9.93.

These drive the algorithm the way the websocket server does, with an
observation whose dimensions have deliberately different scales, and check
that the statistics come to reflect them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig
from plugrl_server.common.data_utils import unbatch_aggregate
from plugrl_server.policy.fpo.fpo_policy import FPOPolicy, FPOPolicyConfig
from plugrl_server.policy.state import slice_policy_step_state

OBS_DIM = 17
# HalfCheetah's measured spread, roughly: small for positions, large for the
# joint velocities at the end.
SCALES = np.array([0.2] * 8 + [1.0] * 3 + [8.0] * 6, dtype=np.float32)
assert SCALES.shape == (OBS_DIM,)


def _algo() -> DPPOAlgorithm:
    torch.manual_seed(0)
    config = DPPOAlgoConfig(
        buffer_size=64,
        batch_size=16,
        update_epochs=1,
        train_itrs=20,
        grad_accum_steps=1,
    )
    algo = DPPOAlgorithm(config, FPOPolicy(FPOPolicyConfig(device="cpu")))
    algo.init_optimizers()
    return algo


def _obs(envs: int, rng: np.random.Generator) -> dict:
    return {
        "states": {
            "obs": rng.normal(0.0, SCALES, size=(envs, OBS_DIM)).astype(np.float32)
        }
    }


def _collect(algo: DPPOAlgorithm, *, envs: int = 2) -> None:
    """Fill the buffer the way websocket_agent_server does."""
    rng = np.random.default_rng(0)
    prev_node = {env: (-1, "") for env in range(envs)}
    terminated = {env: False for env in range(envs)}
    obs = _obs(envs, rng)
    for round_index in range(400):
        _, runtime_state = algo.infer(obs)
        step_state = algo.build_step_state_from_runtime_state(
            runtime_state, include_train_state=True
        )
        next_obs = _obs(envs, rng)
        obs_list = unbatch_aggregate(obs, aggregate_method="concat")
        next_obs_list = unbatch_aggregate(next_obs, aggregate_method="concat")
        done = round_index % 5 == 4
        for env in range(envs):
            one = slice_policy_step_state(step_state, slice(env, env + 1))
            prev_node[env], _, _ = algo.feedback(
                obs=obs_list[env],
                runtime_state=one.runtime_state,
                train_state=one.train_state,
                terminated=terminated[env],
                truncated=False,
                next_obs=next_obs_list[env],
                reward=1.0,
                info={},
                next_terminated=done,
                next_truncated=False,
                prev_node=prev_node[env],
            )
            terminated[env] = done
        obs = next_obs
        if algo.should_learn():
            return
    pytest.fail("the buffer never filled")


def _stats(algo: DPPOAlgorithm) -> dict[str, torch.Tensor]:
    return {
        "count": algo.policy.obs_stats_count.clone(),
        "mean": algo.policy.obs_stats_mean.clone(),
        "std": algo.policy.obs_stats_std.clone(),
    }


class TestTheStatisticsAreMaintained:
    def test_they_start_at_the_identity(self):
        """The state E17 never left."""
        stats = _stats(_algo())

        assert stats["count"].item() == 0.0
        assert torch.equal(stats["mean"], torch.zeros(OBS_DIM))
        assert torch.equal(stats["std"], torch.ones(OBS_DIM))

    def test_one_learn_moves_them_off_the_identity(self):
        algo = _algo()
        _collect(algo)
        algo.pre_learn()
        algo.learn()
        algo.post_learn()

        assert algo.policy.obs_stats_count.item() > 0

    def test_they_come_to_reflect_each_dimension_s_scale(self):
        """The point of the whole thing: 0.2 and 8.0 must not both read as 1."""
        algo = _algo()
        _collect(algo)
        algo.pre_learn()
        algo.learn()
        algo.post_learn()

        std = algo.policy.obs_stats_std
        small = std[:8].mean().item()
        large = std[-6:].mean().item()
        assert small == pytest.approx(0.2, rel=0.3)
        assert large == pytest.approx(8.0, rel=0.3)
        assert large / small > 20


class TestWhenTheyChange:
    def test_they_do_not_move_during_the_learn(self):
        """Old and new log-probabilities must see the same normalisation.

        DPPO's ratio is exp(new - old), and the old log-probability was taken
        at collection under the statistics in force then. If they changed
        before the learn, the ratio would differ from one with no weight
        having moved.
        """
        algo = _algo()
        _collect(algo)
        before = _stats(algo)
        algo.pre_learn()
        algo.learn()
        during = _stats(algo)

        for key in before:
            assert torch.equal(before[key], during[key]), key

    def test_they_move_after_it(self):
        algo = _algo()
        _collect(algo)
        algo.pre_learn()
        algo.learn()
        before = _stats(algo)
        algo.post_learn()

        assert not torch.equal(before["std"], algo.policy.obs_stats_std)


class TestPoliciesWithoutStatistics:
    def test_a_policy_that_keeps_none_is_left_alone(self):
        algo = _algo()
        _collect(algo)
        algo.pre_learn()
        algo.learn()
        # Stand in for a policy with nothing to update.
        algo.policy.update_obs_stats = None  # type: ignore[assignment]

        algo.post_learn()
