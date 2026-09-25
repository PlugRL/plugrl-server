"""Read E20 the way PROTOCOL.md says to, and check its predictions.

`start` is the seed's Phase A value at the checkpoint and `end` the mean of the
arm's last ten iterations, exactly as E16 defined them - FPO samples
deterministically, so Phase A's value is measured the same way as the arms.

Reading order: P1 (the reward arm reproduces E16), P3 (wall clock), P2 (the
zero arm collapses). If P1 fails, P2 is printed but not read.

    python summarise.py [RESULTS_DIR]
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD = "rollout/reward"
WINDOW = 10
EXPECTED_ITERATIONS = 20
REPRODUCE_WITHIN = 1.0
COLLAPSE_BELOW = -500.0
WALL_CLOCK_MINUTES = 60.0

# PROTOCOL.md, known item 1: E16's except-critic arm.
START = {0: 1129.8, 1: 1632.3, 2: 1183.0}
E16_GAIN = {0: 73.0, 1: 366.6, 2: 28.8}

TIME = re.compile(r"^(\d\d:\d\d:\d\d)\|")


def rewards(run: Path) -> list[float]:
    events = sorted(run.glob("tensorboard/events.out.tfevents.*"))
    if not events:
        return []
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if REWARD not in accumulator.Tags().get("scalars", []):
        return []
    return [e.value for e in accumulator.Scalars(REWARD)]


def end_of(curve: list[float]) -> float | None:
    if len(curve) < EXPECTED_ITERATIONS:
        return None
    return statistics.fmean(curve[-WINDOW:])


def stamps(log: Path) -> list[datetime]:
    if not log.exists():
        return []
    return [
        datetime.strptime(m.group(1), "%H:%M:%S")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if (m := TIME.match(line))
    ]


def main() -> int:
    here = Path(__file__).resolve().parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"

    runs: dict[tuple[str, int], Path] = {}
    for run in sorted(root.glob("*/*/*-seed*")):
        arm, _, seed = run.name.rpartition("-seed")
        runs[(arm, int(seed))] = run
    if not runs:
        print(f"no runs under {root}")
        return 1
    end = {key: end_of(rewards(run)) for key, run in runs.items()}

    print(f"P1  the reward arm reproduces E16, within {REPRODUCE_WITHIN}")
    matched, seen = [], []
    for seed in sorted(START):
        value = end.get(("reward", seed))
        if value is None:
            print(f"    seed {seed}: incomplete")
            continue
        seen.append(seed)
        gain = value - START[seed]
        diff = gain - E16_GAIN[seed]
        if abs(diff) <= REPRODUCE_WITHIN:
            matched.append(seed)
        print(
            f"    seed {seed}: gain {gain:+8.2f}   E16 {E16_GAIN[seed]:+8.2f}   difference {diff:+.3f}"
        )
    p1 = bool(seen) and len(matched) == len(seen)
    print(
        f"    {'HOLDS' if p1 else 'FALSIFIED'}  ({len(matched)} of {len(seen)} seeds)"
    )

    print(f"\nP3  wall clock under {WALL_CLOCK_MINUTES:.0f} minutes, all six runs")
    every = [s for log in root.glob("server-*.log") for s in stamps(log)]
    if every:
        span = (max(every) - min(every)).total_seconds() / 60.0
        print(
            f"    {span:.1f} min   {'HOLDS' if span < WALL_CLOCK_MINUTES else 'FALSIFIED'}"
        )

    print(
        f"\nP2  noise advantages collapse a good policy: zero arm below {COLLAPSE_BELOW:.0f} on 2 of 3"
    )
    if not p1:
        print("    P1 failed - printed below, not read as a result")
    rows = [("seed", "start", "reward_gain", "zero_gain")]
    collapsed, complete = [], []
    for seed in sorted(START):
        zero, reward = end.get(("zero", seed)), end.get(("reward", seed))
        if zero is None:
            print(f"    seed {seed}: incomplete")
            continue
        complete.append(seed)
        zero_gain = zero - START[seed]
        reward_gain = (reward - START[seed]) if reward is not None else float("nan")
        if zero_gain < COLLAPSE_BELOW:
            collapsed.append(seed)
        print(
            f"    seed {seed}: start {START[seed]:7.1f}   reward arm {reward_gain:+8.1f}"
            f"   zero arm {zero_gain:+8.1f}"
        )
        rows.append(
            (
                str(seed),
                f"{START[seed]:.1f}",
                f"{reward_gain:+.1f}",
                f"{zero_gain:+.1f}",
            )
        )
    if len(complete) >= 2:
        verdict = "HOLDS" if len(collapsed) >= 2 else "FALSIFIED"
        print(f"    {verdict}  ({len(collapsed)} of {len(complete)} seeds)")
        if verdict == "FALSIFIED" and any(float(r[3]) < 0 for r in rows[1:]):
            print("    the zero arm fell on some seeds by less than 500 - reported,")
            print("    no verdict: the null for this condition was never measured")

    out = here / "summary.tsv"
    out.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
