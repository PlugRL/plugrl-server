# E24 amendments

Dated changes and declared deviations. `PROTOCOL.md` is not edited after the
first registered data point; anything learned afterwards goes here.

---

## 1. Moved off the Windows machine, and rerun from the start

**2026-09-25 18:15, before any cell had finished.** Declared deviation 1: a
run that dies for a reason outside the experiment is restarted once.

The first attempt started at 17:41 on the Windows machine every earlier CPU
experiment used. At 18:06 it was stopped, on the user's request, to free that
machine, which was short of memory with everything else running on it. Wave 1
had been running for 25 minutes; no seed of any cell had finished. Its logs
and checkpoints are kept, untouched and unread, in `results/attempt-1/`.

All six cells rerun from the start on a Linux workstation instead,
`guangzhao` - 24 cores, 125 GB of memory, otherwise idle - under
`~/zuogou/plugrl`, with the same `run.sh`, `run_cell.sh`, seeds, code
(`main` at 8812b54 plus this branch) and settings. What differs, and is
recorded rather than matched:

| | Windows machine | guangzhao |
| --- | --- | --- |
| OS | Windows 11 | Linux |
| torch, numpy | 2.7.1+cpu, 2.3.2 | 2.7.1+cpu, 2.3.2 |
| gymnasium, mujoco (client) | 1.2.3, 3.6.0 | 1.3.0, 3.14.0 |
| dppo | the pinned fork, 89ac416 | the pinned fork, 89ac416 |
| threads per process | torch's default | `OMP_NUM_THREADS=1` |

The thread cap is there because the machine is shared: nine server-client
pairs at torch's default would try to use every core. It changes how long an
iteration takes, not what it computes.

Nothing registered depends on the machine. P1 asks whether each combination
runs end to end, and P2's threshold, 500 against an untrained policy near 0,
is far from anything a change of simulator version could move. The pilot's
null was measured on the Windows machine; the registered runs' own first
iterations are reported beside it.
