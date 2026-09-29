"""Read AMENDMENT.md's six seeds per arm: seeds 0-2 from the registered run,
3-5 from the amendment.

    python amendment.py [results-dir]

Per seed: iterations 91-100's mean, the gain over iteration 1, and the span
from iteration 1 to 100 on the server's clock. Then AMENDMENT.md's test for
a gap worth chasing: all six cross seeds above all six local seeds, or the
arms' six-seed means further apart than the standard deviation within
either arm.
"""

import pathlib
import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
LAYOUT = {("local", s): "local" for s in (0, 1, 2)} | {
    ("cross", s): "cross" for s in (0, 1, 2)
}
LAYOUT |= {("local", s): "local-b" for s in (3, 4, 5)} | {
    ("cross", s): "cross-b" for s in (3, 4, 5)
}


def read(arm: str, seed: int):
    run = (
        R / LAYOUT[(arm, seed)] / "fpo" / "fpo-policy" / f"halfcheetah-{arm}-seed{seed}"
    )
    acc = EventAccumulator(
        str(sorted((run / "tensorboard").glob("events.*"))[-1]),
        size_guidance={"scalars": 0},
    )
    acc.Reload()
    r = acc.Scalars("rollout/reward")
    v = [e.value for e in r]
    return len(v), v[0], statistics.mean(v[90:100]), r[99].wall_time - r[0].wall_time


def main() -> int:
    last, gains, spans = {"local": [], "cross": []}, {"local": [], "cross": []}, {}
    print("seed  arm    iterations  first   last10    gain     span_s")
    for seed in range(6):
        for arm in ("local", "cross"):
            n, first, last10, span = read(arm, seed)
            last[arm].append(last10)
            gains[arm].append(last10 - first)
            spans[(arm, seed)] = span
            print(
                f"{seed:4d}  {arm:5s}  {n:10d}  {first:6.0f}  {last10:7.0f}  {last10 - first:+7.0f}  {span:8.0f}"
            )

    print("\nstatus rule over six seeds (+200 over iteration 1)")
    for arm in ("local", "cross"):
        print(f"    {arm}: {sum(g >= 200 for g in gains[arm])} of 6")

    print("\nextra time, cross minus local, iteration 1 -> 100 (P4 predicted 1,119 s)")
    for seed in range(6):
        print(
            f"    seed {seed}: {spans[('cross', seed)] - spans[('local', seed)]:.0f} s"
        )

    lo, cr = last["local"], last["cross"]
    ml, mc = statistics.mean(lo), statistics.mean(cr)
    sl, sc = statistics.stdev(lo), statistics.stdev(cr)
    separated = min(cr) > max(lo)
    apart = abs(mc - ml) > max(sl, sc)
    print("\niterations 91-100, six seeds per arm")
    print(f"    local mean {ml:.0f}  sd {sl:.0f}  range {min(lo):.0f}-{max(lo):.0f}")
    print(f"    cross mean {mc:.0f}  sd {sc:.0f}  range {min(cr):.0f}-{max(cr):.0f}")
    print(f"    all six cross above all six local: {'yes' if separated else 'no'}")
    print(
        f"    means further apart ({abs(mc - ml):.0f}) than either arm's sd ({max(sl, sc):.0f}): {'yes' if apart else 'no'}"
    )
    print(
        f"    -> {'a gap worth chasing' if separated or apart else 'read as seed variation'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
