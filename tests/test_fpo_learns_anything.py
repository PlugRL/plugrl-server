"""Does this FPO implementation improve a policy on a problem with an answer?

Every FPO test in this repository feeds `reward=1.0`. That checks the
machinery runs - buffers fill, losses are finite, weights move - and it cannot
distinguish learning from drift, because a constant reward carries no signal
about which action was better.

So nothing here has ever shown that FPO makes a policy better at anything.
E14 spent thirteen hours finding that it makes pi0.5 much worse, and five
hypotheses about why have been refuted one hyperparameter at a time. This is
the control those five were missing: a problem small enough to run on a CPU in
seconds, where the right answer is known and the reward says how close the
policy got.

The task is a bandit. The observation is ignored, the reward is the negative
squared distance from the action to a fixed target, and a policy that learns
moves toward the target. If FPO cannot do that here, the defect is in FPO and
not in anything specific to a VLA.
"""

from __future__ import annotations

import numpy as np
import torch

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm
from plugrl_server.common.data_utils import unbatch_aggregate
from plugrl_server.policy.state import slice_policy_step_state
from test_fpo_tree_observations import TreeObsFlowPolicy, _config, _tree_env_obs

TARGET = 0.5
ENVS = 2
# The fixture config runs an 8-transition buffer, which is fine for "does
# the machinery turn" and far too little to tell a policy that is not
# learning from one that has not been given enough to learn from. These
# give the algorithm about 7,700 transitions and still run in seconds.
BUFFER = 256
ITERATIONS = 30


def _algo(**overrides) -> FPOAlgorithm:
    config = _config()
    for key, value in overrides.items():
        setattr(config, key, value)
    algo = FPOAlgorithm(config=config, policy=TreeObsFlowPolicy())
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


def test_fpo_improves_a_policy_on_a_problem_with_a_known_answer():
    """The control every other FPO test in this repository is missing.

    Ten iterations of a bandit whose reward is how close the action got. The
    reward collected in the first iteration comes from the untrained policy;
    the reward in the last comes from a policy that has had nine updates. If
    FPO works at all, the last is higher than the first.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    algo = _algo(buffer_size=BUFFER, batch_size=32, num_updates_per_batch=4)
    rng = np.random.default_rng(0)

    history = [_bandit_iteration(algo, rng) for _ in range(ITERATIONS)]
    first, last = history[0], history[-1]
    best = max(history)

    print("\nmean reward per iteration:")
    for i, r in enumerate(history):
        print(f"  {i + 1:2d}  {r:+.6f}")

    assert np.isfinite(history).all(), f"a reward went non-finite: {history}"
    assert last > first, (
        "FPO did not improve a policy on a bandit with a known answer.\n"
        f"first iteration {first:+.6f}, last {last:+.6f}, best {best:+.6f}.\n"
        "Nothing about a VLA, a batch size of 8 or a trust region is involved "
        "here, so a failure is in FPO itself."
    )
    # Improving and then falling apart is what E14 looks like, and comparing
    # only the ends cannot see it: a run that reaches -0.08 and ends at -0.72
    # still ends above where it started. Rewards are negative, so "within
    # twice the best" means the policy held what it found.
    assert last > 2 * best, (
        "the policy improved and then came apart.\n"
        f"best {best:+.6f} at iteration {history.index(best) + 1}, "
        f"last {last:+.6f}."
    )


class Bf16ActorTreePolicy(TreeObsFlowPolicy):
    """The same toy policy with its actor in bfloat16, as pi0.5's expert is.

    This is the one structural difference between the setting FPO learns in
    above and the setting it collapses in. A bfloat16 actor is what brings
    MasterWeights into play at all: it exists to keep float32 copies of
    half-precision parameters, because an Adam step at a small learning rate
    rounds away in bfloat16. With a float32 policy it has nothing to do.
    """

    def __init__(self) -> None:
        super().__init__()
        self.actor = self.actor.to(torch.bfloat16)

    def _predict_v(self, x, t, cond, *, cond_cache=None) -> torch.Tensor:
        features = self._features(cond, cond_cache, x.shape[0])
        inputs = torch.cat([features, x.flatten(1), t.reshape(-1, 1)], dim=1)
        return self.actor(inputs.to(torch.bfloat16)).float().reshape(x.shape)


def test_a_bfloat16_actor_learns_the_same_bandit():
    """The differential: identical task, identical algorithm, actor in bf16.

    If float32 learns and bfloat16 does not, the defect is in the path only a
    half-precision policy takes - the master weights and the rounding back
    into the model - and not in FPO's objective, its advantages or its trust
    region, all of which are shared.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    config = _config()
    config.buffer_size = BUFFER
    config.batch_size = 32
    config.num_updates_per_batch = 4
    algo = FPOAlgorithm(config=config, policy=Bf16ActorTreePolicy())
    algo.init_optimizers()
    rng = np.random.default_rng(0)

    history = [_bandit_iteration(algo, rng) for _ in range(ITERATIONS)]

    print("\nbfloat16 actor, mean reward per iteration:")
    for i, r in enumerate(history):
        print(f"  {i + 1:2d}  {r:+.6f}")

    best = max(history)
    assert np.isfinite(history).all(), f"a reward went non-finite: {history}"
    assert history[-1] > history[0], (
        "a bfloat16 actor did not learn the bandit that a float32 one does.\n"
        f"first {history[0]:+.6f}, last {history[-1]:+.6f}, best {best:+.6f}."
    )
    assert history[-1] > 2 * best, (
        "a bfloat16 actor learned the bandit and then came apart, where a "
        "float32 one holds what it found.\n"
        f"best {best:+.6f} at iteration {history.index(best) + 1}, "
        f"last {history[-1]:+.6f}.\n"
        "Same task, same algorithm, same hyperparameters - the actor's dtype "
        "is the only difference, and it is what brings MasterWeights into "
        "play."
    )
