"""E58's runs, one line each: env fault dose seed trace [total].

    python jobs.py registered > jobs.txt
    python jobs.py pilot > jobs.txt
    python jobs.py budget > jobs.txt

registered (PROTOCOL.md): Pusher, seeds 0-2, every run traced and ledgered:
the 3 controls, and E54's 21 client faults and 2 log faults at doses one,
0.01 and 1.0. E56's two termination faults are left out: Pusher never
terminates, so they have nothing to change. Seeds 10-19 fault-free, for the
strict check's band.
pilot: seed 9, the controls and every fault at one value.
budget: fault-free seeds 8 and 9, untraced, at 2,048,000 steps, to fix the
budget and the status rule's bar.
"""

from __future__ import annotations

import sys

from faulty_client import CONTROLS, SILENT

LOG = ("log:drop-first", "log:short")
DOSES = ("one", "0.01", "1.0")
NO_TERMINATIONS = ("terminated:drop", "terminated:as-trunc")
FAULTS = tuple(f for f in SILENT if f not in NO_TERMINATIONS) + LOG


def pusher(seed: int, doses=DOSES) -> list[str]:
    jobs = [f"pusher {c} 1.0 {seed} trace" for c in CONTROLS]
    jobs += [f"pusher {f} {d} {seed} trace" for f in FAULTS for d in doses]
    return jobs


def main() -> int:
    which = sys.argv[1]
    if which == "registered":
        jobs = [j for seed in (0, 1, 2) for j in pusher(seed)]
        jobs += [f"pusher none 1.0 {s} notrace" for s in range(10, 20)]
    elif which == "pilot":
        jobs = pusher(9, doses=("one",))
    elif which == "budget":
        jobs = [f"pusher none 1.0 {s} notrace 2048000" for s in (8, 9)]
    else:
        raise SystemExit(f"unknown: {which}")
    print("\n".join(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
