"""Turn the runs' tensorboard scalars into a curve anyone can read.

The event files are binary and each is tens of kilobytes, so they are not
committed; this extracts what the experiment is actually about into a
`summary.tsv` that is, and prints the curve so a reader does not need a
tensorboard server to see the result.

    python summarise.py [RESULTS_DIR]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "rollout/reward"


def seed_curve(events: Path) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(events), size_guidance={"scalars": 0})
    accumulator.Reload()
    if TAG not in accumulator.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in accumulator.Scalars(TAG)]


def collect(results: Path) -> dict[int, list[tuple[int, float]]]:
    curves = {}
    for run in sorted(results.glob("*/*/halfcheetah-seed*")):
        seed = int(run.name.rsplit("seed", 1)[1])
        events = sorted(run.glob("tensorboard/events.out.tfevents.*"))
        if events:
            curves[seed] = seed_curve(events[0])
    return curves


def render(curves: dict[int, list[tuple[int, float]]], width: int = 46) -> str:
    """One row per update, one column block per seed, plus the mean."""
    if not curves:
        return "no curves found"

    every = [v for c in curves.values() for _, v in c]
    lo, hi = min(every), max(every)
    span = (hi - lo) or 1.0

    # Align seeds on the step values the shortest run reached, so the mean is
    # over the same steps for every seed rather than over whatever each one
    # happened to finish.
    steps = sorted(set.intersection(*(set(s for s, _ in c) for c in curves.values())))
    by_seed = {seed: dict(curve) for seed, curve in curves.items()}

    lines = ["step\t" + "\t".join(f"seed{s}" for s in sorted(curves)) + "\tmean"]
    for step in steps:
        values = [by_seed[s][step] for s in sorted(curves)]
        mean = statistics.fmean(values)
        bar = "#" * int((mean - lo) / span * width)
        lines.append(
            f"{step}\t"
            + "\t".join(f"{v:.1f}" for v in values)
            + f"\t{mean:.1f}\t{bar}"
        )
    return "\n".join(lines)


def main() -> int:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    curves = collect(results)
    if not curves:
        print(f"no runs under {results}", file=sys.stderr)
        return 1

    for seed in sorted(curves):
        curve = curves[seed]
        if not curve:
            print(f"seed {seed}: no data yet")
            continue
        print(
            f"seed {seed}: {len(curve)} updates to step {curve[-1][0]}, "
            f"first {curve[0][1]:.1f}, last {curve[-1][1]:.1f}, "
            f"best {max(v for _, v in curve):.1f}"
        )

    table = render({s: c for s, c in curves.items() if c})
    out = results.parent / "summary.tsv"
    # The bar column is for reading here, not for the file.
    out.write_text(
        "\n".join(line.split("\t#")[0].rstrip("\t") for line in table.splitlines())
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
