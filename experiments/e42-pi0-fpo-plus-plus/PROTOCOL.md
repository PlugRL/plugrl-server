# E42 measurement protocol (pre-registered)

**Written 2026-09-28, after the pilot in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E36 ran pi0.5 on LIBERO-10 task 8 under FPO with FPO++'s chunk loss and
per-sample ratio, and it collapsed within five iterations: 5 and 0 of 50,
with the flow-matching loss of its own samples up a thousandfold. That run
kept FPO's own defaults, which FPO++ does not share:
- rewards scaled by 10
- values and advantages recomputed every epoch
- one learning rate for actor and critic
- no gradient clipping

On square the same combination pulled a cloned policy down (E37). With the
rest of FPO++'s fine-tuning (#91), E39 turned that into a rise on every
seed.

**With the rest of FPO++'s fine-tuning, does pi0.5 survive ten iterations of
FPO, and does it improve?**

### What this cannot settle

* Which of the added settings matters. They are added together, as in E39.
* Other tasks. Task 8 only.

---

## Declared in advance: what was already known

1. **The untrained policy** on task 8, fifty episodes, initial states 0-49:
   28-37 over seven evaluations of an unperturbed actor (E15), mean 30.9, sd
   3.6.
2. **E32**: one update with FPO++'s chunk loss, 33 of 50.
3. **E36**, the same arm with FPO's defaults: 5 and 0 of 50 at iteration 5.
   Training success fell to 0 by iterations 6-9. The clipped fraction rose
   to 0.75-0.99, and the flow-matching loss of fresh samples went from 0.11
   to 114 and 407.
4. **E39** on square, with these settings: the critic's explained variance
   reached 0.44-0.48 by iterations 11-20. Every seed rose. The clipped
   fraction stayed at 0.19-0.29.
5. **The pilot** (`results/pilot.txt`).

---

## Design

Two runs on qz103, server seeds 7 and 8. Each gets two cards for the server
(master weights on the second) and its clients on a third. Each runs ten
iterations of 4,096 chunks with a checkpoint after every one.

The settings are E36's `fpopp` arm plus #91's options, as `run.sh` passes
them.
- From E36 (unchanged):
  - chunk loss over the 5 executed steps and the 7 used dimensions
  - velocity error, uniform times, a ratio per sample
  - one critic-only iteration
  - actor learning rate 1e-5
  - 4 epochs of minibatches of 8, 4 samples per action
- Added (FPO++'s):
  - the critic at 1e-4 in its own AdamW group
  - eps 1e-5, weight decay 1e-6, actor beta2 0.99
  - gradients clipped at 25
  - advantages computed once an iteration
  - raw rewards
  - a time-out treated as an ending
  - the Huber error, δ = 1
  - λ 0.99, value loss 0.5 x MSE
  - clip 0.01, FPO++'s square value; E36 used 0.05

Not FPO++'s, and why:
- Advantages are normalised over the buffer, not per minibatch: at
  minibatches of 8, a per-minibatch statistic manufactures signal.
- 4 epochs rather than 10, and 4 samples rather than 8. These are E36's
  values, kept for time and memory; ten FPO iterations already take about 14
  hours.

The harness is E36's with its output under `e42/`, the server on
`$R/plugrl-server-e42` (11ae8be, #91, deployed with CRs stripped), and a
30-hour client timeout. E36's 12 hours killed its tenth iteration.
`derive.py` makes it by exact substitution.

Then each run's iteration-5 and iteration-10 checkpoints are evaluated with
E36's harness: fifty episodes, initial states 0-49 in order, `runner.seed` 7.

---

## Checks

* **V1 - the flags took**: each run's config line shows every setting above.
* **V2 - every evaluation valid**: fifty episodes, client exit 0, `valid`
  true.

---

## The status rule

E36's. A run's iteration-10 checkpoint **learns** at 42 of 50 or more,
**holds** at 20 or more, and **collapses** at 5 or fewer. The coverage
figure's `pi0-policy` · FPO cell learns if both runs do.

---

## Predictions, and what falsifies each

**P1 - both runs complete**: ten iterations, ten checkpoints, no traceback
or out-of-memory.

**P2 - pi0.5 holds at iteration 10 on both runs** (20 or more).

> Grounds: known items 3 and 4. What took square from falling to rising is
> in place, and E36's collapse went with FPO's defaults. Falsified if either
> run is at 19 or fewer.

**Reported, not predicted:**
- whether either run learns. My expectation is that neither does in ten
  iterations of 4,096 chunks.
- success at iterations 5 and 10
- training success, explained variance, clipped fraction and the
  flow-matching loss of fresh samples per iteration
- movement per module group

---

## Declared deviations allowed in advance

1. One restart of any run or evaluation that dies for a reason outside the
   experiment, recorded in `AMENDMENT.md`.
2. Cards may be reassigned if the planned ones are occupied.

---

## Reading order

V1, V2, P1, P2, the status rule, then the reported figures.
