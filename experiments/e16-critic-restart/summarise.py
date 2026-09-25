"""Read E16's runs the way PROTOCOL.md says to read them, and no other way.

The event files are binary and not committed; this extracts what the
experiment is about into `summary.tsv`, which is.

PROTOCOL.md fixes two things this script implements rather than decides:

  * the scalar is `rollout/reward`, the one E6 reported;
  * a run's **value at a step** is the mean over the ten updates ending there,
    because single updates in E6 swung by more than a thousand and no single
    update is a result.

    python summarise.py [PHASE_A_DIR] [PHASE_B_DIR]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "rollout/reward"
WINDOW = 10
FROM_STEP = 327_680
ARMS = ("all", "model", "except-critic")


def curve(run: Path) -> list[tuple[int, float]]:
    events = sorted(run.glob("tensorboard/events.out.tfevents.*"))
    if not events:
        return []
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if TAG not in accumulator.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in accumulator.Scalars(TAG)]


def value_at(points: list[tuple[int, float]], step: int | None = None) -> float | None:
    """The mean of the ten updates ending at `step`, or at the run's end.

    `None` when the run never reached `step`. Without that guard a run still
    in progress reports its latest value under the label of a step it has not
    got to, which is how a partial run gets read as a result.
    """
    if not points:
        return None
    if step is None:
        return statistics.fmean(v for _, v in points[-WINDOW:])
    if points[-1][0] < step:
        return None
    upto = [p for p in points if p[0] <= step]
    if not upto:
        return None
    return statistics.fmean(v for _, v in upto[-WINDOW:])


def phase_a(root: Path) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {}
    for run in sorted(root.glob("*/*/halfcheetah-seed*")):
        out[int(run.name.rsplit("seed", 1)[1])] = curve(run)
    return out


def phase_b(root: Path) -> dict[tuple[int, str], list[tuple[int, float]]]:
    out: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for run in sorted(root.glob("*/*/b-seed*")):
        name = run.name.removeprefix("b-seed")
        seed_text, _, arm = name.partition("-")
        out[(int(seed_text), arm)] = curve(run)
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    a_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"
    b_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "results-b"

    a = phase_a(a_dir)
    b = phase_b(b_dir) if b_dir.exists() else {}

    rows = [
        (
            "phase",
            "seed",
            "arm",
            "first_step",
            "last_step",
            "updates",
            "value_at_327680",
            "value_at_end",
        )
    ]

    print(f"=== Phase A: the control ({a_dir}) ===")
    if not a:
        print("no runs found")
    for seed in sorted(a):
        points = a[seed]
        start = value_at(points, FROM_STEP)
        end = value_at(points)
        first = points[0][0] if points else 0
        last = points[-1][0] if points else 0
        print(
            f"seed {seed}: {len(points):>4} updates, {first}..{last}  "
            f"@{FROM_STEP}={_fmt(start)}  end={_fmt(end)}  "
            f"{_arrow(start, end)}"
        )
        rows.append(
            (
                "A",
                seed,
                "-",
                first,
                last,
                len(points),
                _tsv(start),
                _tsv(end),
            )
        )

    if b:
        print(f"\n=== Phase B: three arms from step {FROM_STEP} ({b_dir}) ===")
        print(f"{'seed':>4} {'arm':<14} {'updates':>8} {'start':>9} {'end':>9}  held?")
        for seed in sorted({s for s, _ in b}):
            # Every arm starts from the same place: this seed's Phase A value
            # at the checkpoint step. An arm's own first logged update is
            # already one update of its own training, so it is not the start.
            start = value_at(a.get(seed, []), FROM_STEP)
            for arm in ARMS:
                points = b.get((seed, arm))
                if points is None:
                    continue
                end = value_at(points)
                first = points[0][0] if points else 0
                last = points[-1][0] if points else 0
                print(
                    f"{seed:>4} {arm:<14} {len(points):>8} "
                    f"{_fmt(start):>9} {_fmt(end):>9}  {_arrow(start, end)}"
                )
                rows.append(
                    ("B", seed, arm, first, last, len(points), _tsv(start), _tsv(end))
                )

    out = here / "summary.tsv"
    out.write_text(
        "\n".join("\t".join(str(c) for c in row) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} ({len(rows) - 1} rows)")
    return 0


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _tsv(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _arrow(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return ""
    return "rose" if end > start else "FELL"


if __name__ == "__main__":
    raise SystemExit(main())
