"""Is there a dtype effect at all, or was it seed 0?

The bfloat16 collapse - best -0.079, final -0.715 - was measured on one seed
against one float32 run on the same seed. Three seeds of bfloat16 then showed
the collapse on seed 0 only: seeds 1 and 2 ended at 1.17x and 1.82x their
best, which is not a collapse.

So the float32 control has to be run on the same three seeds. If float32
holds everywhere and bfloat16 fails somewhere, the effect is weak but real.
If float32 also comes apart on some seed, there is no dtype effect and the
localisation claimed yesterday does not stand.
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


def run(policy_cls, seed, iterations=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = _config()
    config.buffer_size = BUFFER
    config.batch_size = 32
    config.num_updates_per_batch = 4
    algo = FPOAlgorithm(config=config, policy=policy_cls())
    algo.init_optimizers()
    rng = np.random.default_rng(seed)
    rewards = [_bandit_iteration(algo, rng) for _ in range(iterations)]
    return max(rewards), rewards[-1]


def main():
    print(f"{'seed':>4} {'dtype':>9} {'best':>9} {'final':>9} {'final/best':>11}")
    summary = {"float32": [], "bfloat16": []}
    for seed in (0, 1, 2, 3, 4):
        for name, cls in (("float32", TreeObsFlowPolicy), ("bfloat16", Bf16ActorTreePolicy)):
            best, final = run(cls, seed)
            ratio = final / best if best else float("nan")
            summary[name].append(ratio)
            print(f"{seed:4d} {name:>9} {best:+9.4f} {final:+9.4f} {ratio:11.2f}")

    print("\nfinal/best across seeds (1.0 means it held what it found):")
    for name, ratios in summary.items():
        worst = max(ratios)
        print(
            f"  {name:>9}  " + " ".join(f"{r:5.2f}" for r in ratios)
            + f"   worst {worst:5.2f}"
        )


if __name__ == "__main__":
    main()
