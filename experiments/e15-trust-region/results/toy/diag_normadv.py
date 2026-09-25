"""Does advantage normalisation cause the policy to stop holding its solution?

The mechanism is written in the code that normalises them, as the thing that
change does not fix: normalising rescales whatever spread it finds to 1, so a
buffer holding only noise comes out at unit scale. A converged policy is
exactly that case - every action earns about the same reward, so the true
advantages go to zero and the division turns them back into full-magnitude
"this one was better" signals that are noise.

That predicts the shape actually measured: learning works while the policy is
far from the optimum and there is real spread in the returns, and the policy
comes apart once it arrives, because the updates stop carrying information
and do not stop.

One flag decides it. Five seeds, both precisions, normalisation on and off.
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

from test_fpo_learns_anything import (  # noqa: E402
    ITERATIONS,
    POLICIES,
    SEEDS,
    _algo,
    _bandit_iteration,
)


def run(dtype, seed, normalize):
    torch.manual_seed(seed)
    np.random.seed(seed)
    algo = _algo(POLICIES[dtype], normalize_advantage=normalize)
    rng = np.random.default_rng(seed)
    rewards = [_bandit_iteration(algo, rng) for _ in range(ITERATIONS)]
    return max(rewards), rewards[-1]


def main():
    print(
        f"{'dtype':>9} {'seed':>4} | {'normalise on':>22} | "
        f"{'normalise off':>22}"
    )
    print(f"{'':9} {'':4} | {'best':>9} {'final':>6} {'x':>4} | "
          f"{'best':>9} {'final':>6} {'x':>4}")
    worst = {True: 0.0, False: 0.0}
    for dtype in sorted(POLICIES):
        for seed in SEEDS:
            row = f"{dtype:>9} {seed:4d} |"
            for normalize in (True, False):
                best, final = run(dtype, seed, normalize)
                ratio = final / best if best else float("nan")
                worst[normalize] = max(worst[normalize], ratio)
                row += f" {best:+9.4f} {final:+6.3f} {ratio:4.1f} |"
            print(row)

    print(
        f"\nworst final/best across all ten runs: "
        f"normalise on {worst[True]:.2f}, off {worst[False]:.2f}"
    )


if __name__ == "__main__":
    main()
