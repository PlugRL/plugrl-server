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

---

## 2. Training here is not deterministic, and two statements above rested on it

**2026-09-25, written after E18's results were read and after E20 measured
what this corrects.** Nothing in P1's or P2's verdict changes; what changes is
what else the protocol and amendment 1 claimed.

The protocol's known item 3 says one-client training is deterministic on this
machine, so "an E18 seed and the E17 seed of the same number differ only
through the code". Its evidence was that two runs' **first** iterations
matched to the decimal. E20 then ran one configuration twice (#52, its
amendment 1): the runs agree until the first update and then drift apart, by
21% after a single update on one seed and by up to 85 in twenty-iteration
gain. **Collection is deterministic; learning is not.** The cause was not
isolated. Parallel CPU reductions summed in an order that depends on thread
scheduling are the usual one.

Three things built on that premise are withdrawn:

1. **The per-seed pairing of E18 against E17.** The two do not differ only
   through the code. `FINDINGS.md` compares them as two independent groups of
   three instead - Welch's t on the gains - and says why.
2. **Amendment 1's "contention changes how long an iteration takes, not what
   it computes".** If the drift comes from scheduling, contention is exactly
   the kind of thing that could change what is computed. So E18's data is not
   shown to be what an idle machine would have produced. It is data from a
   machine that was shared for part of the run, which amendment 1 records.
3. **Known item 2's "about 4% different after [the first update]"** as a
   measure of the accumulation fix. A 4% difference after one update is
   inside what two identical runs produce. The same correction was made on
   #49.

**What does not depend on it.** P1 reads `obs_stats_count` and the spread of
`obs_stats_std` at the final checkpoint: a count of every observation the run
saw, and statistics of them. That is structural, not a trajectory, and holds
3 of 3. P2 is an absolute threshold on each seed's own gain, +200. It never
compared E18 with E17. The largest gain was +20.2, 180 short, and even
twenty-iteration noise of FPO's size, up to 85, does not close that gap. P2
stays falsified, 0 of 3.
