"""Are the bfloat16 run's weights bad, or is its forward pass bad?

float32 and bfloat16 actors walk almost the same path in weight space - norms
1.24 to 4.27 against 1.25 to 4.66 - and their rewards separate completely
from iteration 22, ending at -0.053 and -0.715. Similar weights, very
different behaviour, which puts the difference in the forward pass rather
than in the optimizer or the weights themselves.

This separates the two. Train in bfloat16 for 30 iterations, then measure the
same weights twice: once through the bfloat16 forward the policy has been
using, and once cast to float32 and run again. Nothing about the weights
changes between the two measurements.

  both bad      the weights are damaged, and precision in the forward is not
                the story
  f32 recovers  the weights are fine and the bfloat16 forward is what loses
                the policy - which is testable on pi0.5 the same way
"""

import copy
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(
    0,
    "D:/75128/Desktop/plugrl-work/plugrl-server/.claude/worktrees/"
    "fix-pi0-image-mask-batching/tests",
)

from plugrl_server.algorithm.fpo.fpo import FPOAlgorithm  # noqa: E402
from test_fpo_learns_anything import (  # noqa: E402
    BUFFER,
    ENVS,
    TARGET,
    Bf16ActorTreePolicy,
    _bandit_iteration,
    _config,
)
from test_fpo_tree_observations import _tree_env_obs  # noqa: E402


def measure(policy, rng, rounds=200) -> float:
    """Mean reward of the policy as it stands, with no learning."""
    rewards = []
    obs = _tree_env_obs(ENVS, rng)
    for _ in range(rounds):
        action, _ = policy.get_action_and_runtime_state(obs)
        taken = np.asarray(action, dtype=np.float32).reshape(ENVS, -1)
        for env in range(ENVS):
            rewards.append(-float(((taken[env] - TARGET) ** 2).mean()))
        obs = _tree_env_obs(ENVS, rng)
    return float(np.mean(rewards))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    config = _config()
    config.buffer_size = BUFFER
    config.batch_size = 32
    config.num_updates_per_batch = 4
    algo = FPOAlgorithm(config=config, policy=Bf16ActorTreePolicy())
    algo.init_optimizers()
    rng = np.random.default_rng(0)

    for _ in range(30):
        _bandit_iteration(algo, rng)

    trained = algo.policy

    torch.manual_seed(1)
    bf16_score = measure(trained, np.random.default_rng(1))

    # Same weights, float32 forward. The subclass casts inputs to bfloat16 in
    # _predict_v, so use the float32 parent's method with a float32 actor.
    from test_fpo_tree_observations import TreeObsFlowPolicy

    f32 = TreeObsFlowPolicy()
    f32.actor = copy.deepcopy(trained.actor).to(torch.float32)
    f32.critic = copy.deepcopy(trained.critic)

    torch.manual_seed(1)
    f32_score = measure(f32, np.random.default_rng(1))

    print(f"trained 30 iterations with a bfloat16 actor")
    print(f"  measured through the bfloat16 forward : {bf16_score:+.6f}")
    print(f"  same weights, float32 forward         : {f32_score:+.6f}")


if __name__ == "__main__":
    main()
