"""Read E18 the way PROTOCOL.md says to, and check its predictions.

Reading order is P1, then P3, then P2. P1 is a manipulation check read from
each seed's final checkpoint: if the observation statistics are not live in
the run, the run is E17 again and P2 is not read at all.

A run's value at a point is the mean of the ten iterations ending there;
`start` at the tenth, `best` the highest, `end` the last - as E17.

    python summarise.py [RESULTS_DIR]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import safetensors.torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD = "rollout/reward"
WINDOW = 10
EXPECTED_ITERATIONS = 100
LEARNED_BY = 200.0
SPREAD_AT_LEAST = 10.0

# E17, the null: the same design without #50. From E17's summary.tsv.
NULL = {
    0: (-291.1, -281.1, -312.3),
    1: (-313.1, -281.1, -321.5),
    2: (-289.6, -266.5, -318.1),
}


def rewards(run: Path) -> list[float]:
    events = sorted(run.glob("tensorboard/events.out.tfevents.*"))
    if not events:
        return []
    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if REWARD not in accumulator.Tags().get("scalars", []):
        return []
    return [e.value for e in accumulator.Scalars(REWARD)]


def final_checkpoint(run: Path) -> Path | None:
    steps = sorted(
        (int(p.name) for p in run.iterdir() if p.is_dir() and p.name.isdigit()),
    )
    return run / str(steps[-1]) if steps else None


def rolling(values: list[float]) -> list[float]:
    return [
        statistics.fmean(values[max(0, i - WINDOW + 1) : i + 1])
        for i in range(len(values))
    ]


def main() -> int:
    here = Path(__file__).resolve().parent
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "results"

    seeds = {}
    for run in sorted(root.glob("*/*/halfcheetah-seed*")):
        seeds[int(run.name.rsplit("seed", 1)[1])] = run
    if not seeds:
        print(f"no runs under {root}")
        return 1

    print("P1  the fix took effect - obs_stats at the final checkpoint")
    live = []
    for seed, run in sorted(seeds.items()):
        ck = final_checkpoint(run)
        if ck is None:
            print(f"    seed {seed}: no checkpoint")
            continue
        state = safetensors.torch.load_file(str(ck / "model.safetensors"))
        count = float(state["obs_stats_count"])
        std = state["obs_stats_std"]
        spread = float(std.max() / std.min())
        ok = count > 0 and spread > SPREAD_AT_LEAST
        if ok:
            live.append(seed)
        print(
            f"    seed {seed}: step {ck.name}  count {count:>10.0f}  "
            f"std {float(std.min()):.2f}..{float(std.max()):.2f}  "
            f"spread {spread:>5.1f}x  {'live' if ok else 'NOT LIVE'}"
        )
    p1 = len(live) == len(seeds) and len(seeds) > 0
    print(f"    {'HOLDS' if p1 else 'FALSIFIED'}  ({len(live)} of {len(seeds)} seeds)")
    if not p1:
        print("\nP1 failed: the run did not test what it was built to test.")
        print("PROTOCOL.md reads nothing further.")
        return 0

    print("\nP3  wall clock is read from the server logs, not from here.")

    print(f"\nP2  DPPO learns: end - start > {LEARNED_BY:.0f} on at least 2 of 3")
    header = f"    {'seed':>4} {'itrs':>5} {'start':>9} {'best':>9} {'end':>9} {'gain':>8}   null gain"
    print(header)
    learned, complete = [], []
    rows = [("seed", "iterations", "start", "best", "end", "gain", "null_gain")]
    for seed, run in sorted(seeds.items()):
        values = rewards(run)
        if len(values) < EXPECTED_ITERATIONS:
            print(
                f"    {seed:>4} {len(values):>5}   incomplete - no verdict read from it"
            )
            continue
        complete.append(seed)
        window = rolling(values)
        start, best, end = window[WINDOW - 1], max(window), window[-1]
        null_start, _, null_end = NULL[seed]
        if end - start > LEARNED_BY:
            learned.append(seed)
        print(
            f"    {seed:>4} {len(values):>5} {start:>9.1f} {best:>9.1f} {end:>9.1f} "
            f"{end - start:>+8.1f}   {null_end - null_start:>+8.1f}"
        )
        rows.append(
            (
                str(seed),
                str(len(values)),
                f"{start:.1f}",
                f"{best:.1f}",
                f"{end:.1f}",
                f"{end - start:+.1f}",
                f"{null_end - null_start:+.1f}",
            )
        )
    if len(complete) < 2:
        print("    fewer than two complete seeds - PROTOCOL.md reads nothing from this")
        return 0
    print(
        f"    {'HOLDS' if len(learned) >= 2 else 'FALSIFIED'}"
        f"  ({len(learned)} of {len(complete)} seeds)"
    )

    out = here / "summary.tsv"
    out.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
