"""FPO learns a bandit and then does not hold it.

Every other FPO test in this repository feeds `reward=1.0`. That checks the
machinery turns - buffers fill, losses are finite, weights move - and cannot
tell learning from drift, because a constant reward carries no signal about
which action was better. So nothing here had ever shown FPO makes a policy
better at anything, or keeps it better.

The task below is a bandit on a two-layer flow policy: the reward is the
negative squared distance from the action to a fixed target, so the answer is
known and the reward says how close the policy got. It runs on a CPU in about
a minute.

Two things come out of it, and the second is the defect:

  FPO learns.        From about -1.5 to about -0.06 within twenty iterations,
                     on every seed and in both precisions.
  FPO does not hold. On most seeds the policy then degrades, by up to nine
                     times its best, while still being trained on the same
                     reward.

That is E14's shape - a pi0.5 policy went from 29 of 50 to 0 of 50 across one
iteration and stayed there - reproduced with no VLA, no half precision
required, no batch size forced by memory and no trust region subtlety.

**On dtype.** An earlier version of this file asserted that a bfloat16 actor
collapses where a float32 one holds, on the strength of one seed where the
bfloat16 run ended nine times worse than its best. Five seeds do not support
that. float32 ends 4.59x worse than its best on seed 4 and 2.72x on seed 2,
and on seeds 2, 3 and 4 the bfloat16 run held *better* than the float32 one.
There is no dtype effect here; the instability is in the algorithm as
configured, and the bfloat16 case is kept as the control that shows so.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.common.data_utils import unbatch_aggregate
from plugrl_server.policy.state import slice_policy_step_state
from test_fpo_tree_observations import TreeObsFlowPolicy, _config, _tree_env_obs

TARGET = 0.5
ENVS = 2
# The fixture config runs an 8-transition buffer, which is fine for "does the
# machinery turn" and far too little to tell a policy that is not learning
# from one that has not been given enough to learn from. These give the
# algorithm about 7,700 transitions and still run in seconds.
BUFFER = 256
ITERATIONS = 30
SEEDS = (0, 1, 2, 3, 4)


class Bf16ActorTreePolicy(TreeObsFlowPolicy):
    """The same toy policy with its actor in bfloat16, as pi0.5's expert is.

    Kept as a control. It brings MasterWeights into play - that class exists
    to hold float32 copies of half-precision parameters and has nothing to do
    for a float32 policy - and the measurements show it changes nothing about
    whether the policy holds what it learns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.actor = self.actor.to(torch.bfloat16)

    def _predict_v(self, x, t, cond, *, cond_cache=None) -> torch.Tensor:
        features = self._features(cond, cond_cache, x.shape[0])
        inputs = torch.cat([features, x.flatten(1), t.reshape(-1, 1)], dim=1)
        return self.actor(inputs.to(torch.bfloat16)).float().reshape(x.shape)


POLICIES = {"float32": TreeObsFlowPolicy, "bfloat16": Bf16ActorTreePolicy}


def _algo(policy_cls, **overrides) -> FPOAlgorithm:
    config = _config()
    config.buffer_size = BUFFER
    config.batch_size = 32
    config.num_updates_per_batch = 4
    for key, value in overrides.items():
        setattr(config, key, value)
    algo = FPOAlgorithm(config=config, policy=policy_cls())
    algo.init_optimizers()
    return algo


def _bandit_iteration(algo: FPOAlgorithm, rng) -> float:
    """Fill the buffer once, rewarding closeness to TARGET. Returns mean reward."""
    prev_node = {env: (-1, "") for env in range(ENVS)}
    terminated = {env: False for env in range(ENVS)}
    obs = _tree_env_obs(ENVS, rng)
    rewards: list[float] = []

    for round_index in range(400):
        action, runtime_state = algo.infer(obs)
        step_state = algo.build_step_state_from_runtime_state(
            runtime_state, include_train_state=True
        )
        next_obs = _tree_env_obs(ENVS, rng)
        obs_list = unbatch_aggregate(obs, aggregate_method="concat")
        next_obs_list = unbatch_aggregate(next_obs, aggregate_method="concat")
        done = round_index % 3 == 2
        taken = np.asarray(action, dtype=np.float32).reshape(ENVS, -1)

        for env in range(ENVS):
            reward = -float(((taken[env] - TARGET) ** 2).mean())
            rewards.append(reward)
            one = slice_policy_step_state(step_state, slice(env, env + 1))
            prev_node[env], _, _ = algo.feedback(
                obs=obs_list[env],
                runtime_state=one.runtime_state,
                train_state=one.train_state,
                terminated=terminated[env],
                truncated=False,
                next_obs=next_obs_list[env],
                reward=reward,
                info={},
                next_terminated=done,
                next_truncated=False,
                prev_node=prev_node[env],
            )
            terminated[env] = done
        obs = next_obs
        if algo.should_learn():
            break

    assert algo.should_learn(), "the buffer never filled"
    algo.pre_learn()
    algo.learn()
    algo.post_learn()
    return float(np.mean(rewards))


def _history(dtype: str, seed: int) -> list[float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    algo = _algo(POLICIES[dtype])
    rng = np.random.default_rng(seed)
    return [_bandit_iteration(algo, rng) for _ in range(ITERATIONS)]


@pytest.mark.parametrize("dtype", sorted(POLICIES))
def test_fpo_learns_a_bandit_with_a_known_answer(dtype: str):
    """It does learn, and this has never been shown in this repository before."""
    history = _history(dtype, seed=0)
    assert np.isfinite(history).all(), f"a reward went non-finite: {history}"
    assert max(history) > 5 * history[0], (
        "FPO did not improve a policy on a bandit with a known answer.\n"
        f"first {history[0]:+.6f}, best {max(history):+.6f}."
    )


@pytest.mark.parametrize("dtype", sorted(POLICIES))
@pytest.mark.parametrize("seed", SEEDS)
def test_fpo_holds_what_it_learns(dtype: str, seed: int):
    """It does not, and that is the defect. Committed failing.

    Rewards are negative, so "final within twice the best" is a loose bar: a
    policy that reached -0.06 may end at -0.12 and still pass. Measured
    final/best over thirty iterations on five seeds:

        float32   1.05  1.01  2.72  1.70  4.59
        bfloat16  9.00  1.17  1.82  1.50  2.76

    Comparing only the first and last iterations cannot see this - a run that
    reaches -0.08 and ends at -0.72 still ends above where it started.
    """
    history = _history(dtype, seed)
    best, final = max(history), history[-1]
    assert np.isfinite(history).all(), f"a reward went non-finite: {history}"
    assert final > 2 * best, (
        "the policy improved and then came apart while still being trained on "
        "the same reward.\n"
        f"best {best:+.6f} at iteration {history.index(best) + 1}, "
        f"final {final:+.6f}, ratio {final / best:.2f}x."
    )
