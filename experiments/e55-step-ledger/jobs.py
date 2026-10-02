"""E55's same-machine runs, one line each: env fault dose seed trace.

    python jobs.py registered > jobs.txt
    python jobs.py pilot > jobs.txt

registered (PROTOCOL.md): Pendulum, seeds 0-2, every run traced and
ledgered: the 3 controls; E54's 21 client faults and 2 log faults at doses
one, 0.01 and 1.0. pilot: the same at seed 9. The runs across machines are
cross.sh's.
"""

from __future__ import annotations

import sys

from faulty_client import CONTROLS, SILENT

LOG = ("log:drop-first", "log:short")
DOSES = ("one", "0.01", "1.0")


def pendulum(seed: int) -> list[str]:
    jobs = [f"pendulum {c} 1.0 {seed} trace" for c in CONTROLS]
    jobs += [f"pendulum {f} {d} {seed} trace" for f in SILENT + LOG for d in DOSES]
    return jobs


def main() -> int:
    which = sys.argv[1]
    if which == "registered":
        jobs = [j for seed in (0, 1, 2) for j in pendulum(seed)]
    elif which == "pilot":
        jobs = pendulum(9)
    else:
        raise SystemExit(f"unknown: {which}")
    print("\n".join(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
