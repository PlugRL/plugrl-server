"""E56's runs, one line each: env fault dose seed trace.

    python jobs.py registered > jobs.txt
    python jobs.py pilot > jobs.txt

registered (PROTOCOL.md): Hopper, seeds 0-2, every run traced and ledgered:
the 3 controls, and the 23 client faults (E54's 21 and E56's two
termination faults) and 2 log faults at doses one, 0.01 and 1.0. Seeds
10-19 fault-free, for the strict check's band. pilot: seed 9, the controls
and every fault at one cell.
"""

from __future__ import annotations

import sys

from faulty_client import CONTROLS, SILENT

LOG = ("log:drop-first", "log:short")
DOSES = ("one", "0.01", "1.0")


def hopper(seed: int, doses=DOSES) -> list[str]:
    jobs = [f"hopper {c} 1.0 {seed} trace" for c in CONTROLS]
    jobs += [f"hopper {f} {d} {seed} trace" for f in SILENT + LOG for d in doses]
    return jobs


def main() -> int:
    which = sys.argv[1]
    if which == "registered":
        jobs = [j for seed in (0, 1, 2) for j in hopper(seed)]
        jobs += [f"hopper none 1.0 {s} notrace" for s in range(10, 20)]
    elif which == "pilot":
        jobs = hopper(9, doses=("one",))
    else:
        raise SystemExit(f"unknown: {which}")
    print("\n".join(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
