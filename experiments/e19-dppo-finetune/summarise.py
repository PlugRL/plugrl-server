"""Read E19 the way PROTOCOL.md says to, and check its predictions.

`start` is a seed's FIRST iteration under DPPO - collected before any DPPO
update, so it measures the restored policy under DPPO's noisy sampling,
untouched. `end` is the mean of the last ten iterations. Phase A's value at
the checkpoint was measured under FPO's deterministic sampling and is printed
for reference only; using it as the start would count the cost of DPPO's
sampling noise as something DPPO did to the policy.

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
EXPECTED_ITERATIONS = 20
TAIL = 10
BEHAVES_TRAINED_BY = 500.0
COLLAPSE_BELOW = -500.0
WALL_CLOCK_MINUTES = 45.0

# PROTOCOL.md, known items 1 and 2.
RANDOM_FIRST_ITERATION = {0: -356.0, 1: -356.8, 2: -290.2}
PHASE_A_AT_CHECKPOINT = {0: 1129.8, 1: 1632.3, 2: 1183.0}
E16_FPO_EXCEPT_CRITIC = {0: 73.0, 1: 366.6, 2: 28.8}

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


def minutes(log: Path) -> float | None:
    stamps = [
        datetime.strptime(m.group(1), "%H:%M:%S")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if (m := TIME.match(line))
    ]
    if len(stamps) < 2:
        return None
    return (stamps[-1] - stamps[0]).total_seconds() / 60.0


def main() -> int:
    here = Path(__file__).resolve().parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"

    runs = {
        int(run.name.rsplit("seed", 1)[1]): run
        for run in sorted(root.glob("*/*/halfcheetah-seed*"))
    }
    if not runs:
        print(f"no runs under {root}")
        return 1

    print("P1(a) the restore took effect, mechanically")
    loaded = []
    for seed in sorted(runs):
        log = (root / f"server-seed{seed}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        restored = "restore=except-critic" in log
        held = "Holding back 12 critic tensors" in log
        if restored and held:
            loaded.append(seed)
        print(
            f"      seed {seed}: restore logged {restored}, 12 critic tensors held back {held}"
        )
    if len(loaded) != len(runs):
        print("      FALSIFIED - the run did not load a policy; nothing below is read")
        return 0
    print("      HOLDS")

    curves = {seed: rewards(run) for seed, run in runs.items()}
    complete = [s for s, c in curves.items() if len(c) >= EXPECTED_ITERATIONS]
    for seed, curve in sorted(curves.items()):
        if len(curve) < EXPECTED_ITERATIONS:
            print(
                f"\n      seed {seed}: {len(curve)}/{EXPECTED_ITERATIONS} iterations - incomplete"
            )
    if len(complete) < 2:
        print("\nfewer than two complete seeds - PROTOCOL.md reads nothing from this")
        return 0

    print(
        f"\nP1(b) the restored policy acts trained: first iteration beats random by > {BEHAVES_TRAINED_BY:.0f}"
    )
    trained = []
    for seed in complete:
        first = curves[seed][0]
        margin = first - RANDOM_FIRST_ITERATION[seed]
        if margin > BEHAVES_TRAINED_BY:
            trained.append(seed)
        print(
            f"      seed {seed}: first {first:8.1f}   random {RANDOM_FIRST_ITERATION[seed]:7.1f}"
            f"   margin {margin:+8.1f}   (Phase A, FPO sampling: {PHASE_A_AT_CHECKPOINT[seed]:.1f})"
        )
    p1b = len(trained) == len(complete)
    print(
        f"      {'HOLDS' if p1b else 'FALSIFIED - DPPO sampling makes the policy act near random'}"
    )

    print(f"\nP3    wall clock under {WALL_CLOCK_MINUTES:.0f} minutes")
    worst = 0.0
    for seed in sorted(runs):
        m = minutes(root / f"server-seed{seed}.log")
        if m is not None:
            worst = max(worst, m)
            print(f"      seed {seed}: {m:5.1f} min")
    print(f"      {'HOLDS' if worst < WALL_CLOCK_MINUTES else 'FALSIFIED'}")

    print(f"\nP2    no collapse: end - start > {COLLAPSE_BELOW:.0f} on every seed")
    if not p1b:
        print("      read under P1(b)'s caveat: the start is already near random")
    rows = [("seed", "start", "end", "gain", "e16_fpo_gain", "phase_a")]
    collapsed = []
    for seed in complete:
        curve = curves[seed]
        start = curve[0]
        end = statistics.fmean(curve[-TAIL:])
        gain = end - start
        if gain <= COLLAPSE_BELOW:
            collapsed.append(seed)
        print(
            f"      seed {seed}: start {start:8.1f}   end {end:8.1f}   gain {gain:+8.1f}"
            f"   | E16 FPO except-critic gain {E16_FPO_EXCEPT_CRITIC[seed]:+7.1f} (reference, no verdict)"
        )
        rows.append(
            (
                str(seed),
                f"{start:.1f}",
                f"{end:.1f}",
                f"{gain:+.1f}",
                f"{E16_FPO_EXCEPT_CRITIC[seed]:+.1f}",
                f"{PHASE_A_AT_CHECKPOINT[seed]:.1f}",
            )
        )
    print(
        f"      {'HOLDS' if not collapsed else f'FALSIFIED - seeds {collapsed} collapsed'}"
    )

    out = here / "summary.tsv"
    out.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
