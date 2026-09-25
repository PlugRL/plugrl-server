"""Does Phase A still reproduce E6, fourteen days and a lot of merges later?

Phase A is E6's configuration on today's `main`, which is the whole reason it
exists: E6 ran on `cb5b369` on 2026-09-10, and reading a 2026-09-24 result
against E6's published numbers would confound "the code changed" with
whatever the experiment is actually testing.

This prints both curves at E6's own reported steps so the reproduction can be
checked rather than assumed. E6's column is transcribed from the per-seed
table in `../e6-first-learning-curve/FINDINGS.md`, which reports the single
update nearest each step - not the mean of ten that `summarise.py` uses - so
this script reads single updates too. They are not comparable to
`summary.tsv`'s numbers and are not meant to be.

E6's own document says single updates swing by more than a thousand after
step 100,000, which is the size of difference to expect here.

    python compare_e6.py [RESULTS_DIR]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "rollout/reward"

# ../e6-first-learning-curve/FINDINGS.md, "the curve" table: seeds 0, 1, 2.
E6: dict[int, tuple[float, float, float]] = {
    4_096: (-303.9, -296.4, -344.2),
    40_960: (-102.4, -96.1, -194.5),
    102_400: (696.8, 393.8, 64.5),
    204_800: (1268.9, 1497.5, 777.2),
    307_200: (1789.8, 1049.1, 850.2),
    409_600: (2210.5, 2033.9, 1481.6),
}


def curve(run: Path) -> dict[int, float]:
    events = sorted(run.glob("tensorboard/events.out.tfevents.*"))
    if not events:
        return {}
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if TAG not in accumulator.Tags().get("scalars", []):
        return {}
    return {e.step: e.value for e in accumulator.Scalars(TAG)}


def main() -> int:
    here = Path(__file__).resolve().parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"

    curves = {}
    for run in sorted(root.glob("*/*/halfcheetah-seed*")):
        seed = int(run.name.rsplit("seed", 1)[1])
        points = curve(run)
        if points:
            curves[seed] = points
    if not curves:
        print(f"no runs under {root}")
        return 1

    print(f"{'step':>8} {'E6':>9} {'Phase A':>9}   per-seed Phase A")
    rows = [("step", "e6_mean", "phase_a_mean", "seed0", "seed1", "seed2")]
    for step, published in E6.items():
        # The nearest logged update, because a save or a partial buffer can
        # put an update a step or two off the round number - Phase A's seeds
        # wrote checkpoints at 81921 and 327681 for exactly that reason.
        got = [
            points[min(points, key=lambda s: abs(s - step))]
            for points in curves.values()
        ]
        print(
            f"{step:>8} {statistics.fmean(published):>9.1f} "
            f"{statistics.fmean(got):>9.1f}   " + " ".join(f"{v:8.1f}" for v in got)
        )
        rows.append(
            (
                str(step),
                f"{statistics.fmean(published):.1f}",
                f"{statistics.fmean(got):.1f}",
                *[f"{v:.1f}" for v in got],
            )
        )

    out = here / "e6-comparison.tsv"
    out.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
