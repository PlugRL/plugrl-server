"""If the value loss is prevented from diverging, does the policy hold?

Across ten runs the reward's fall from its best and the value loss's rise
from its lowest agree on a two-times bar eight times out of ten, with a rank
correlation of 0.988. The worst collapse, 9.0x, sits with the worst
divergence, 55.5x; the two steadiest runs, 1.01x and 1.05x, sit with 1.00x
and 1.34x.

That is an association, not a direction. This tests the direction the cheap
way: value_loss_coeff scales how hard the critic is pushed, and if the
critic's divergence is what takes the policy down, pushing it less should
show up in both numbers at once.

Default is 0.25. Runs 0.25, 0.05 and 0.01 on the five seeds in float32.
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


def run(seed, coeff):
    torch.manual_seed(seed)
    np.random.seed(seed)
    algo = _algo(POLICIES["float32"], value_loss_coeff=coeff)
    rng = np.random.default_rng(seed)
    rewards, vlosses = [], []
    for _ in range(ITERATIONS):
        reward, m = iteration_with_metrics(algo, rng)
        rewards.append(reward)
        vlosses.append(m["losses"]["value_loss"])
    return max(rewards), rewards[-1], min(vlosses), vlosses[-1]


def main():
    coeffs = (0.25, 0.05, 0.01)
    print(f"{'seed':>4} | " + " | ".join(f"{'coeff ' + str(c):>26}" for c in coeffs))
    print(f"{'':4} | " + " | ".join(
        f"{'best':>8} {'final':>7} {'fall':>4} {'vrise':>4}" for _ in coeffs
    ))
    worst = {c: 0.0 for c in coeffs}
    for seed in SEEDS:
        row = f"{seed:4d} |"
        for c in coeffs:
            best, final, vmin, vfin = run(seed, c)
            fall = final / best
            rise = vfin / vmin if vmin else float("nan")
            worst[c] = max(worst[c], fall)
            row += f" {best:+8.4f} {final:+7.3f} {fall:4.1f} {rise:4.1f} |"
        print(row)

    print("\nworst reward fall across the five seeds:")
    for c in coeffs:
        print(f"  value_loss_coeff {c:<5} {worst[c]:.2f}x")


if __name__ == "__main__":
    main()
