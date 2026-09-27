# E25 measurement protocol (pre-registered)

**Written 2026-09-25, after the pilots in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

The platform has trained pi0.5 with one algorithm, FPO, and DPPO has a
`libero` variant written for exactly this setting, which had never run. **Does
pi0.5 train end to end under DPPO on this platform, and does DPPO's update
destroy it the way FPO's first iteration does?**

The second half matters because of E19 and E22. DPPO barely moved trained
HalfCheetah policies - 0.28% to 0.43% in twenty updates, against FPO's 11% to
16% - and FPO's destruction of pi0.5 lives in the direction its update pushes
the expert's MLP (E22). An algorithm whose updates are that small should not
destroy the policy in two iterations.

### What this cannot settle

* Whether DPPO *improves* pi0.5. Two iterations cannot show it, and nothing
  about improvement is registered.
* Anything about DPPO against FPO beyond this setting: minibatch, buffer and
  learning rate are each algorithm's own.

---

## Declared in advance: what was already known

1. **The pilots** (`results/pilot.txt`): the first found a platform defect,
   fixed in #58 before anything else ran; the second ran out of memory at
   the libero variant's minibatch of 128; the third, at minibatch 8, ran two
   iterations of 512 to the end, peaking at 21,317 MiB on a 24 GB card.
2. **The null** (E15 correction 2): an unperturbed actor, 28 to 37 of 50,
   30.9 ± 3.6, on the older code. E26 is measuring it on newer code in
   parallel; E25's threshold does not depend on the difference.
3. **FPO's first iteration**: 0 of 50 (E14), after moving the expert 1.51%.

---

## Design

`pi0-policy default dppo libero`, `pi05_libero`, on `main` at 8812b54 plus #58
(`$R/plugrl-server-main`), with everything the variant sets except:

* **buffer 4,096** entries, E14's per-iteration amount of data;
* **minibatch 8**, from pilot 2, with the variant's 16 accumulation steps;
* **two iterations**, `--algo.train-itrs 2`, a checkpoint after each. The
  variant has no learning-rate scheduler, so no warmup bounds the length.

Server on cards 0 and 1 (policy on the first), ten LIBERO clients on card 2,
task 8, randomised initial states, replanning every 5 steps, seed 7 - E14's
client. Then the last checkpoint evaluated with E14's harness (`eval.sh`,
derived from `e14_eval.sh` by `derive.py`): task 8, fifty episodes, initial
states 0 to 49, `runner.seed` 7. `movement.py` measures the checkpoint's
distance from the base per module group.

---

## Predictions, and what falsifies each

**P1 - pi0.5 trains end to end under DPPO.** Two iterations logged, a
checkpoint after each, no traceback, the server exits on its own, and card 0
stays under its 24 GB.

> Falsified by any part failing.

**P2 - DPPO's two iterations do not destroy the policy.** The last
checkpoint scores **at least 20 of 50**.

> Read only if P1 holds. 20 is three standard deviations below the null's
> mean. Holds: DPPO's updates to pi0.5, like its updates to HalfCheetah
> policies, are too small to do FPO's damage in two iterations. Falsified at
> 5 or less: DPPO destroys it too, and the damage is not FPO's alone.
> 6 to 19: reported, no reading.

**Reported, not predicted:** the movement per module group, beside FPO's.

---

## Declared deviations allowed in advance

1. One restart of a run or evaluation that dies for a reason outside the
   experiment, recorded in `AMENDMENT.md`.

---

## Reading order

P1, then P2, then the movements.
