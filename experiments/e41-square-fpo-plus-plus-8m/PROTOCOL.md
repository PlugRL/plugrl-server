# E41 measurement protocol (pre-registered)

**Written 2026-09-28, after E39's result and the pilot in
`results/pilot.txt`, before the registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question, and why it is asked now

E39 ran `fpo-policy` on square under FPO with FPO++'s square fine-tuning for
4.8M steps. Every seed rose through the whole run. One seed passed the bar,
and the other two fell short by 0.04 and 0.01. FPO++'s own square runs are
8M steps (arXiv 2602.02481, the experiments on robomimic).

**Run on to FPO++'s 8M steps, does E39's setting learn square?**

This question was decided after seeing E39's result, and it is declared as
such. What keeps it from being a search is what is fixed here and taken
from outside:
- The budget is FPO++'s published one, 8M steps: 167 iterations of 48,000,
  or 8.016M.
- The start and the bar are E39's, unchanged.
- There is one extension. There will be no further one on this cell under
  this setting.

---

## Declared in advance: what was already known

1. **E39**: gains of +0.292, +0.160 and +0.193 at iteration 100.
   - Success over iterations 51-100 by window of ten: 0.701, 0.713, 0.774,
     0.759, 0.787 / 0.660, 0.657, 0.644, 0.687, 0.684 / 0.670, 0.712,
     0.692, 0.661, 0.711.
   - Fifty-episode evaluations of the final checkpoints: 0.80, 0.64, 0.64,
     against the clone's 0.50.
2. **The pilot** (`results/pilot.txt`): seed 0 resumed from E39's final
   checkpoint with `--algo.restore all`. The server logged "Restoring from
   ... with restore=all", and the first iteration it logged was 101.
3. **A resume is not an uninterrupted run.** The seeds' random streams
   restart. The episodes in progress when E39 stopped are not continued,
   and the first iteration's buffer is collected fresh. The weights, both
   AdamW groups' moments, the step count and the iteration count carry
   over.

---

## Design

The E39 cell, resumed: each seed from its own E39 final checkpoint
(1200003, 1200003, 1200002) with `--algo.restore all`, run until the step
count reaches 167 x 12,000. Every setting is E39's, with the same three
client processes per seed, on `guangzhao`. Code: this branch (#91, E39,
and this directory).

---

## Checks

* **V1 - it is a resume**: every seed's server log restores its own E39
  seed with `restore=all` and shows E39's settings, and its first logged
  iteration is 101. A seed that fails is not read.

---

## The status rule

A seed's start is E39's: its mean `rollout/success` over E39's iterations
1-2. Its end is the mean over iterations 158-167. The cell **learns** if the
end exceeds the start by at least **+0.2** on at least **2 of 3** seeds.

---

## Predictions, and what falsifies each

**P1 - the resumed cell runs end to end**: 67 iterations logged, at least 6
checkpoints, no traceback, clients exit 0, on every seed.

**P2 - it learns within 8M steps.**

> Grounds: known item 1. Over E39's last forty iterations seed 2 rose about
> 0.04 and seed 1 about 0.02. Seed 2 is 0.007 short, and at that rate it
> passes within a few iterations. Seed 1 is 0.04 short and would need most
> of the 67, and it has been flat since iteration 60. Seed 0 is already past
> by 0.09. Two seeds past means seed 0 has to stay past and seed 2 has to
> cross. Falsified if fewer than two seeds end at least 0.2 above their
> start.

**Reported, not predicted:**
- success by window across both runs
- fifty-episode evaluations of the final checkpoints, with E39's `eval50.sh`
- wall clock

---

## Declared deviations allowed in advance

1. One restart of any seed that dies for a reason outside the experiment,
   from the same E39 checkpoint, recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, then the reported figures.
