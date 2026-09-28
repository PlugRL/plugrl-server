# E36 measurement protocol (pre-registered)

**Written 2026-09-27, before any E36 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

pi0.5 on LIBERO-10 task 8 now survives its first update under both
algorithms PlugRL runs on it: FPO with FPO++'s chunk loss and per-sample
ratio (E32, 33 of 50) and DPPO's `libero` variant (E25, 30 of 50 after two
iterations). Neither has been shown to make it better.

**Run ten iterations, does either improve the policy?**

### What this cannot settle

* Whether more data per update, or more updates, would. Every setting is the
  one E32 and E25 ran; only the number of iterations changes.
* Other tasks. Task 8 only.

---

## Declared in advance: what was already known

1. **The untrained policy** on task 8, fifty episodes, initial states 0 to
   49: 28 to 37 over seven evaluations of an unperturbed actor (E15), mean
   30.9, standard deviation 3.6; 29 in E26; 36 in E32.
2. **E32's `fpopp`**: one critic-only iteration, then one update: 33 of 50;
   the expert moved 1.64%.
3. **E25's DPPO**: two iterations, 30 of 50; it moved the expert's MLP and
   attention about 2% as far as one FPO iteration.
4. **An iteration is 4,096 environment steps**, a few dozen episodes of up to
   520 steps across ten clients; an FPO iteration takes about 78 minutes and a
   DPPO one about an hour on these cards.

---

## Design

Three runs on qz103, ten iterations each, a checkpoint after every one:

| run | what | cards | server seed |
| --- | --- | --- | --- |
| `fpopp-s7` | E32's `fpopp` unchanged: FPO with FPO++'s chunk loss and per-sample ratio, the first iteration critic-only, then nine updates | 0, 1 | 7 |
| `fpopp-s8` | the same | 2, 3 | 8 |
| `dppo` | E25's cell unchanged: `pi0-policy default dppo libero`, buffer 4,096, minibatch 8 | 4, 5 | 7 |

Everything else is E32's and E25's: `pi05_libero`, ten LIBERO clients on task
8 with randomised initial states, replanning every 5 steps, client seed 7.
Then each run's iteration-5 and iteration-10 checkpoints evaluated with E32's
harness: fifty episodes, initial states 0 to 49 in order, `runner.seed` 7.
Harnesses derived from E32's `train.sh` and `eval.sh` and E25's `cell.sh` by
`derive.py` (output directories, the server seed, the code directory, and
E25's memory recorder following its cards). Code: #74 (5272832), deployed LF
as `$R/plugrl-server-e32`, which has #58.

---

## Checks

* **V1 - every run's flags took**: the config lines show E32's `fpopp`
  settings for the two FPO runs and the `libero` variant with buffer 4,096
  and minibatch 8 for DPPO.
* **V2 - every evaluation valid**: fifty episodes, client exit 0, `valid`
  true in the harness's table.

---

## The status rule

A run's iteration-10 checkpoint **learns** at **42 of 50** or more - above
the untrained policy's mean by more than three of its standard deviations,
and five above the best it has ever scored. It **holds** at 20 or more and
**collapses** at 5 or fewer. The coverage figure's cell learns if every one
of its runs does: both `fpopp` runs for `pi0-policy` · FPO, the one DPPO run
for `pi0-policy` · DPPO.

---

## Predictions, and what falsifies each

**P1 - all three runs complete**: ten iterations, ten checkpoints, no
traceback or out-of-memory.

**P2 - FPO++ keeps pi0.5 through nine updates**: both `fpopp` runs hold at
iteration 10.

> Grounds: known item 2. Falsified if either run is at 19 or fewer.

**P3 - DPPO keeps it through ten iterations**: `dppo` holds at iteration 10.

> Grounds: known item 3 - it barely moves the policy. Falsified at 19 or
> fewer.

**Reported, not predicted:** whether any run learns - my expectation is
that none does in ten iterations of this little data; every run's success
at iterations 5 and 10; the training rollouts' success per iteration; the
movement per module group at iterations 5 and 10.

---

## Declared deviations allowed in advance

1. One restart of any run or evaluation that dies for a reason outside the
   experiment, recorded in `AMENDMENT.md`.
2. Cards may be reassigned if the planned ones are occupied.

---

## Reading order

V1, V2, P1, P2, P3, the status rule, then the reported figures.
