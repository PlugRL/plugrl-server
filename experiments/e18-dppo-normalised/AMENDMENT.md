# E18 amendments

Dated changes and declared deviations. `PROTOCOL.md` is not edited after the
first registered data point; anything learned afterwards goes here.

---

## 1. The machine stopped being the experiment's alone

**2026-09-25 11:50, written while E18 was running and before any of its
results were read.**

E18 inherits E17's condition of three runs at a time **alone on the machine**.
From 11:47 that stopped being true. Between 11:39 and 11:49 the three seeds
advanced four iterations, against about sixteen in the ten minutes before.
There was no suspension - Kernel-Power recorded nothing in that window - but
the process table showed two processes that were not E18's:

| PID | started | what | load |
| --- | --- | --- | --- |
| 36404 | 11:47:25 | `meme_vs_sticker.py`, from a separate Claude session on a different project | about 1,327 CPU-seconds in its first two minutes, roughly eleven cores |
| 26252 | 11:27:23 | that project's own server | 2.1 GB resident |

Neither was started by this experiment or this session, and neither was
stopped: they belong to someone else's work.

**What it does not change.** One-client training is deterministic on this
machine - E18's first iteration matched E17's to the decimal on all three
seeds - and contention changes how long an iteration takes, not what it
computes. So P1 and P2 read the same data they would have read on an idle
machine.

**What it does change.** P3, the wall-clock prediction, no longer measures
E18's cost. It is still computed and printed, and it is **not read as a
result**: a bound that another project's workload can break is not a bound
on this one.
