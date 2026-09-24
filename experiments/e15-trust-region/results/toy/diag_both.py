"""Does the float32 actor's weight norm grow the way the bfloat16 one's does?

The bfloat16 diagnostic showed the rounding residue staying at the
quantisation level throughout - master and model never drift apart - while
the actor's weight norm grew monotonically from 1.25 to 4.66. The reward
improved up to about 3.8 and fell after it.

So the question is no longer about precision. It is whether the growth is
something bfloat16 does and float32 does not, or something both do with only
one of them minding. Same task, same seeds, same everything but the dtype.
"""

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
    Bf16ActorTreePolicy,
    _bandit_iteration,
    _config,
)
from test_fpo_tree_observations import TreeObsFlowPolicy  # noqa: E402


def actor_norm(algo) -> float:
    """Works for both: a float32 actor has no master copies to look at."""
    return float(
        torch.cat(
            [p.detach().float().flatten() for p in algo.policy.actor.parameters()]
        ).norm()
    )


def run(label, policy_cls, iterations=30):
    torch.manual_seed(0)
    np.random.seed(0)
    config = _config()
    config.buffer_size = BUFFER
    config.batch_size = 32
    config.num_updates_per_batch = 4
    algo = FPOAlgorithm(config=config, policy=policy_cls())
    algo.init_optimizers()
    rng = np.random.default_rng(0)

    rows = []
    for _ in range(iterations):
        reward = _bandit_iteration(algo, rng)
        rows.append((reward, actor_norm(algo)))
    return label, rows


def main():
    results = [
        run("float32", TreeObsFlowPolicy),
        run("bfloat16", Bf16ActorTreePolicy),
    ]
    print(f"{'it':>3}  {'f32 reward':>11} {'f32 norm':>9}   "
          f"{'bf16 reward':>11} {'bf16 norm':>9}")
    for i in range(len(results[0][1])):
        (fr, fn) = results[0][1][i]
        (br, bn) = results[1][1][i]
        print(f"{i + 1:3d}  {fr:+11.6f} {fn:9.4f}   {br:+11.6f} {bn:9.4f}")


if __name__ == "__main__":
    main()
