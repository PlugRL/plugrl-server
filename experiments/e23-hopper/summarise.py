"""Read E23's verdicts from the runs' own logs and tensorboards.

    python summarise.py

P1, then P3, then P2, then the descriptive figures - PROTOCOL.md's reading
order. `results/run.out` is run.sh's own output, which holds the client exit
codes and the start and end times.
"""

import datetime
import pathlib
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = RESULTS / "fpo" / "fpo-policy"
SEEDS = (0, 1, 2)
STEPS = 409_600


def curve(seed: int, tag: str) -> list[float]:
    events = sorted((RUNS / f"hopper-seed{seed}" / "tensorboard").glob("events.*"))
    if not events:
        return []
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return []
    return [e.value for e in acc.Scalars(tag)]


def main() -> int:
    run_out = (RESULTS / "run.out").read_text(encoding="utf-8", errors="replace")
    exits = {
        int(s): int(rc) for s, rc in re.findall(r"seed (\d) finished rc=(\d+)", run_out)
    }

    print("P1  the platform completes the runs")
    p1 = True
    for seed in SEEDS:
        reward = curve(seed, "rollout/reward")
        steps = [
            int(p.name)
            for p in (RUNS / f"hopper-seed{seed}").iterdir()
            if p.name.isdigit()
        ]
        near = any(abs(s - STEPS) <= 4096 for s in steps)
        log = (RESULTS / f"server-seed{seed}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        traceback = "Traceback" in log
        ok = len(reward) >= 100 and near and not traceback and exits.get(seed) == 0
        p1 &= ok
        print(
            f"    seed {seed}: {len(reward)} iterations, checkpoint near {STEPS}: {near}, "
            f"traceback: {traceback}, client rc {exits.get(seed)}  {'ok' if ok else 'FAILS'}"
        )
    print(f"    {'HOLDS' if p1 else 'FALSIFIED'}")

    print("\nP3  wall clock under 150 minutes")
    stamps = re.findall(r"^(start|end)\s+(\S+ \S+)$", run_out, flags=re.M)
    times = {k: datetime.datetime.strptime(v, "%Y-%m-%d %H:%M:%S") for k, v in stamps}
    if "start" in times and "end" in times:
        minutes = (times["end"] - times["start"]).total_seconds() / 60
        print(f"    {minutes:.1f} minutes  {'HOLDS' if minutes < 150 else 'FALSIFIED'}")
    else:
        print("    not read - no start and end in run.out")

    print("\nP2  FPO learns Hopper: end >= 500 on at least 2 of 3")
    if not p1:
        print("    not read - P1 does not hold")
        return 0
    passed = 0
    rows = []
    for seed in SEEDS:
        reward = curve(seed, "rollout/reward")
        length = curve(seed, "rollout/length")
        start, end = reward[0], sum(reward[90:100]) / 10
        end_len = sum(length[90:100]) / 10
        passed += end >= 500
        rows.append((seed, start, end, end - start, length[0], end_len, max(reward)))
        print(
            f"    seed {seed}: start {start:8.1f}   end {end:8.1f}   {'>= 500' if end >= 500 else '< 500'}"
        )
    print(f"    {'HOLDS' if passed >= 2 else 'FALSIFIED'}  ({passed} of 3 seeds)")

    print("\nreported: gain, episode length, best iteration")
    print(
        "    seed    start      end     gain   length at start   length at end   best"
    )
    for seed, start, end, gain, len0, len_end, best in rows:
        print(
            f"    {seed:4d} {start:8.1f} {end:8.1f} {gain:+8.1f}   {len0:15.1f}   {len_end:13.1f} {best:8.1f}"
        )
    over = [r[0] for r in rows if r[2] >= 1000]
    print(f"    seeds whose end is 1,000 or more: {over or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
