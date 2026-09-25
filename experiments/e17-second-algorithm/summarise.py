"""Read E17 the way PROTOCOL.md says to read it, and check its predictions.

The reading order is the one the protocol fixes: P4, then P3, then P1, then
P2. P3 comes before P1 because a run whose critic never learned cannot support
a claim about what its policy did.

A run's **value at a point** is the mean over the ten iterations ending there,
as E16 and E6 define it. `start` is that mean at the tenth iteration, `best`
the highest it reaches, `end` its value at the last.

    python summarise.py [RESULTS_DIR]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD = "rollout/reward"
EXPLAINED = "rollout/explained_variance"
WINDOW = 10
EXPECTED_ITERATIONS = 100
# P2's ratio is taken on value + SHIFT so that a ratio of returns straddling
# zero means something. PROTOCOL.md fixes the 400.
SHIFT = 400.0
LEARNED_BY = 500.0
CRITIC_BAR = 0.5


def scalars(run: Path, tag: str) -> list[float]:
    events = sorted(run.glob("tensorboard/events.out.tfevents.*"))
    if not events:
        return []
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [e.value for e in accumulator.Scalars(tag)]


def rolling(values: list[float]) -> list[float]:
    return [
        statistics.fmean(values[max(0, i - WINDOW + 1) : i + 1])
        for i in range(len(values))
    ]


def main() -> int:
    here = Path(__file__).resolve().parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"

    runs = {}
    for run in sorted(root.glob("*/*/halfcheetah-seed*")):
        seed = int(run.name.rsplit("seed", 1)[1])
        reward = scalars(run, REWARD)
        if reward:
            runs[seed] = (reward, scalars(run, EXPLAINED))
    if not runs:
        print(f"no runs under {root}")
        return 1

    rows = [
        ("seed", "iterations", "start", "best", "end", "gain", "explained_variance")
    ]
    print(
        f"{'seed':>4} {'itrs':>5} {'start':>9} {'best':>9} {'end':>9} {'gain':>8} {'expl_var':>9}"
    )
    learned, critics, held, complete = [], [], [], []
    for seed, (reward, explained) in sorted(runs.items()):
        window = rolling(reward)
        if len(reward) < EXPECTED_ITERATIONS:
            print(f"{seed:>4} {len(reward):>5}   incomplete - no verdict read from it")
            continue
        complete.append(seed)
        start, best, end = window[WINDOW - 1], max(window), window[-1]
        variance = explained[-1] if explained else float("nan")
        if end - start > LEARNED_BY:
            learned.append(seed)
        if variance > CRITIC_BAR:
            critics.append(seed)
        if (best + SHIFT) <= 2 * (end + SHIFT):
            held.append(seed)
        print(
            f"{seed:>4} {len(reward):>5} {start:>9.1f} {best:>9.1f} {end:>9.1f} "
            f"{end - start:>+8.1f} {variance:>9.3f}"
        )
        rows.append(
            (
                str(seed),
                str(len(reward)),
                f"{start:.1f}",
                f"{best:.1f}",
                f"{end:.1f}",
                f"{end - start:+.1f}",
                f"{variance:.3f}",
            )
        )

    n = len(complete)
    print()
    if n < 2:
        print("fewer than two complete seeds - PROTOCOL.md reads nothing from this")
        return 0
    print(
        f"P3  critic learns, explained_variance > {CRITIC_BAR}"
        f"   {'HOLDS' if len(critics) >= 2 else 'FALSIFIED'}"
        f"  ({len(critics)} of {n} seeds)"
    )
    print(
        f"P1  DPPO learns, end - start > {LEARNED_BY:.0f}"
        f"        {'HOLDS' if len(learned) >= 2 else 'FALSIFIED'}"
        f"  ({len(learned)} of {n} seeds)"
    )
    print(
        f"P2  holds what it learns, within 2x of best  "
        f"{'HOLDS' if len(held) >= 2 else 'FALSIFIED'}"
        f"  ({len(held)} of {n} seeds)"
    )
    if len(learned) < 2:
        print("    P2 is vacuous here: a policy that learned almost nothing")
        print("    cannot lose much of it.")
    print("\nP4 is wall clock and is read from the server logs, not from here.")

    out = here / "summary.tsv"
    out.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
