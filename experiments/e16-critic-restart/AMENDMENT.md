# E16 amendments

Dated changes and declared deviations. `PROTOCOL.md` is not edited after the
first registered data point; anything learned afterwards goes here.

---

## 1. The checkpoint step is not the number the save interval implies

**2026-09-24 20:09, written after Phase A finished and before Phase B ran.**

`phase_b.sh` as pre-registered hard-coded `FROM_STEP=327680`. Phase A's
checkpoints landed at:

| seed | checkpoints |
| --- | --- |
| 0 | 81921, 163840, 245760, **327680**, 409600 |
| 1 | 81920, 163840, 245760, **327681**, 409600 |
| 2 | 81920, 163840, 245760, **327681**, 409600 |

A checkpoint directory is named for the step the save happened at, and where
in the collection loop that falls varies by a step. Seed 0 is one step late at
its first checkpoint and exact at its fourth; seeds 1 and 2 are the other way
round.

So the registered harness would have found no checkpoint for **two of three
seeds** and the experiment would have quietly run on one seed.

`phase_b.sh` now takes the nearest checkpoint to `FROM_STEP`, refuses anything
further off than one update (4,096 steps), prints which step it used when that
is not the one asked for, and measures the `all` arm's step budget from the
step the chosen checkpoint actually carries. Committed at `e4e19d0`, before
Phase B started. The run log records `seed 1: using step 327681 for 327680`
and the same for seed 2.

This changes no prediction. It is the difference between reading P1-P5 on
three seeds and reading them on one.

## 2. The machine suspended for 32 minutes during Phase B

**2026-09-24 21:20, written while Phase B was still running.**

Batch 2 (seed 1) started at 20:25 and had done 18 of its 20 updates at 21:14,
against batch 1's 20 updates in 15 minutes. The cause is not the experiment:

```
20:41:52  Kernel-Power 506  system entering Modern Standby
21:14:02  Kernel-Power 507  system exiting Modern Standby
```

which matches the gap in the client's own timing log exactly - the last line
before it is 20:41:52 and the next is 21:14:01.

**The runs survived it and the data is unaffected.** All three of batch 2's
servers report zero disconnects, zero reconnects and zero timeouts across the
suspension: the whole machine froze, timers included, so nothing timed out.
Collection and learning resumed where they stopped.

Two consequences, both recorded rather than worked around:

* **Phase B's wall clocks are not a cost measurement** and are not reported as
  one. `PROTOCOL.md` registers a wall-clock prediction for Phase A only (P6,
  which held at 91 minutes against a 150-minute bound), so no prediction is
  affected.
* Later batches may suspend the same way. The power setting was left alone:
  changing it is a system configuration change nobody asked for, and the
  measured effect is on elapsed time rather than on the result.

This is the same failure mode E8 documented, where a suspension of 1 h 53 min
was first blamed on a WebSocket keepalive and then shown not to be - learn
steps of 190 s, nine times the ping timeout, close nothing. A frozen machine
does not drop connections either, and the zero-disconnect count above is that
finding holding a second time.
