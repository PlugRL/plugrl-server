"""Does a diverging value loss predict the policy losing what it learned?

Two seeds suggested it. On seed 1, which holds, the value loss falls
monotonically to 0.24. On seed 4, which comes apart, it bottoms at 0.64 and
rises fourteen-fold to 9.02 while the reward falls from -0.15 to -0.65. Over
the same stretch the flow-matching loss keeps falling and the policy ratio
stays at 0.99 on both, so the critic is the only thing behaving differently.

Two points is not an association. This runs all five seeds in both precisions
and reports, for each, how far the reward fell from its best and how far the
value loss rose from its lowest. If the two move together across ten runs the
signal is real and measurable during training, which is what an early stop
would need.
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diag_turn import iteration_with_metrics  # noqa: E402
from test_fpo_learns_anything import ITERATIONS, POLICIES, SEEDS, _algo  # noqa: E402


def run(dtype, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    algo = _algo(POLICIES[dtype])
    rng = np.random.default_rng(seed)
    rewards, vlosses = [], []
    for _ in range(ITERATIONS):
        reward, m = iteration_with_metrics(algo, rng)
        rewards.append(reward)
        vlosses.append(m["losses"]["value_loss"])
    return rewards, vlosses


def main():
    print(
        f"{'dtype':>9} {'seed':>4} {'reward fall':>12} {'vloss rise':>11}  verdict"
    )
    pairs = []
    for dtype in sorted(POLICIES):
        for seed in SEEDS:
            rewards, vlosses = run(dtype, seed)
            fall = rewards[-1] / max(rewards)
            rise = vlosses[-1] / min(vlosses)
            pairs.append((fall, rise))
            collapsed = fall > 2.0
            diverged = rise > 2.0
            verdict = (
                "both" if collapsed and diverged
                else "neither" if not collapsed and not diverged
                else "reward only" if collapsed
                else "vloss only"
            )
            print(
                f"{dtype:>9} {seed:4d} {fall:12.2f} {rise:11.2f}  {verdict}"
            )

    falls = np.array([p[0] for p in pairs])
    rises = np.array([p[1] for p in pairs])
    agree = int(((falls > 2) == (rises > 2)).sum())
    print(f"\nagreement on the 2x bar: {agree} of {len(pairs)} runs")
    print(f"spearman-ish (rank correlation of the two): "
          f"{np.corrcoef(falls.argsort().argsort(), rises.argsort().argsort())[0, 1]:.3f}")


if __name__ == "__main__":
    main()
