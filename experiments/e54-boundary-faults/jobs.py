"""E54's runs, one line each: env fault dose seed trace.

    python jobs.py registered > jobs.txt
    python jobs.py pilot > jobs.txt

registered (PROTOCOL.md):
- pendulum, seeds 0-2: the 3 controls; the 23 silent faults (21 the
  client's, 2 the log's) at the 5 doses; the 2 loud faults at one cell. All
  traced.
- pendulum, seeds 10-19: fault-free, for the strict curve check.
- halfcheetah, seeds 0-2: none; the 23 silent faults at doses one, 0.01, 1.0.
- halfcheetah, seeds 10-19: fault-free.
pilot: pendulum at seed 9 as registered; halfcheetah at seed 9 for none,
reward:stale at one cell and obs:f16 at 0.01.
"""

from __future__ import annotations

import sys

from faulty_client import CONTROLS, LOUD, SILENT

LOG = ("log:drop-first", "log:short")
DOSES = ("one", "0.001", "0.01", "0.1", "1.0")
CHEETAH_DOSES = ("one", "0.01", "1.0")


def pendulum(seed: int) -> list[str]:
    jobs = [f"pendulum {c} 1.0 {seed} trace" for c in CONTROLS]
    jobs += [f"pendulum {f} {d} {seed} trace" for f in SILENT + LOG for d in DOSES]
    jobs += [f"pendulum {f} one {seed} trace" for f in LOUD]
    return jobs


def main() -> int:
    which = sys.argv[1]
    if which == "registered":
        jobs = []
        for seed in (0, 1, 2):
            jobs += pendulum(seed)
        jobs += [f"pendulum none 1.0 {s} notrace" for s in range(10, 20)]
        for seed in (0, 1, 2):
            jobs.append(f"halfcheetah none 1.0 {seed} notrace")
            jobs += [
                f"halfcheetah {f} {d} {seed} notrace"
                for f in SILENT + LOG
                for d in CHEETAH_DOSES
            ]
        jobs += [f"halfcheetah none 1.0 {s} notrace" for s in range(10, 20)]
    elif which == "pilot":
        jobs = pendulum(9) + [
            "halfcheetah none 1.0 9 notrace",
            "halfcheetah reward:stale one 9 notrace",
            "halfcheetah obs:f16 0.01 9 notrace",
        ]
    else:
        raise SystemExit(f"unknown: {which}")
    # HalfCheetah's runs are four times longer: start them first.
    jobs.sort(key=lambda j: not j.startswith("halfcheetah"))
    print("\n".join(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
