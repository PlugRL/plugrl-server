# E26 measurement protocol (pre-registered)

**Written 2026-09-25, before any E26 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E22 (#57) located the damage one FPO iteration does to pi0.5 in its update
to the action expert's MLP. FPO's first update restricted to the expert's 54
MLP matrices takes the policy to 0 of 50; restricted to attention, the adaRMS
modulation or the projections, it leaves 24 to 26. That is a statement about
a perturbation built after the fact. The version that could become a remedy
is a statement about training:

**If FPO trains pi0.5 with the expert's MLP frozen, does its first iteration
still destroy the policy?**

#60 adds `--policy.freeze-expert-params`, which freezes every expert
parameter whose name matches a pattern; `\.mlp\.` matches exactly the 54 MLP
tensors, leaving 154 of the actor's 208 trainable tensors, 203.6M of 430.1M
parameters (checked on the real pi0.5 before this file was written).

### Why a control, and why a fresh baseline

The code is today's `main` plus #60, and much has changed since E14 ran: the
advantage-normalisation scope (#41), critic warmup (#42), the drift stop
(#45), resume (#39). E15 showed the collapse survives #41, but nothing has
shown it on this exact code. So the unfrozen control runs beside the frozen
arm, on the same code and harness, and the untrained policy is evaluated on
the same code as well: the null of 30.9 ± 3.6 was measured through the older
copy E14 ran.

### What this cannot settle

* One iteration. A frozen MLP that survives the first iteration may still
  fail later, or may not improve anything. Surviving is the claim; improving
  is not.
* That the MLP is *necessary* in the strict sense: freezing it also changes
  the gradients the other groups receive. The frozen arm is a remedy test,
  not a counterfactual of E14's own update.
* One task, one seed, one evaluation per arm.

---

## Declared in advance: what was already known

1. **The null** (E15 correction 2): an unperturbed actor on the older code,
   seven evaluations, 28 to 37, 30.9 ± 3.6.
2. **FPO's first iteration** on the older code: 0 of 50 (E14), and 0 for
   every E15 arm that moved the expert 0.66% to 1.51%.
3. **E22's localisation**: FPO's first update restricted to `mlp` 0; to
   `attn` 26; to `mod` 25; to `io` 24.
4. **The freeze on the real policy**: 208 → 154 trainable actor tensors with
   `\.mlp\.`, none of the expert's MLP tensors trainable (#60's check).

---

## Design

Three arms on qz103, E14's settings throughout: `pi0-policy` `pi05_libero`,
FPO with learning rate 1e-5, 4 updates per batch, clip 0.05, buffer 4,096,
minibatch 8, 4 samples per action, float32 master weights on the second
server card; ten LIBERO clients on task 8 with randomised initial states,
replanning every 5 steps, seed 7; one iteration.

| arm | trained | evaluated |
| --- | --- | --- |
| `base` | no | the untrained policy |
| `control` | one FPO iteration, nothing frozen | its iteration-1 checkpoint |
| `frozen` | one FPO iteration, `--policy.freeze-expert-params '\.mlp\.'` | its iteration-1 checkpoint |

Every evaluation is E14's: task 8, fifty episodes, initial states 0 to 49 in
order, `runner.seed` 7. `train.sh` and `eval.sh` are E14's harnesses, derived
from `e14_train.sh` and `e14_eval.sh` by `derive.py`, which changes only the
code directory (through `PYTHONPATH`), the output directory and the freeze
flag, and refuses to write if any substitution misses. The code is `main` at
8812b54 plus #60 (4d7ee38), deployed LF as `$R/plugrl-server-e26`.

---

## Checks

* **V1 - the freeze held.** In `frozen`'s checkpoint all 54 expert MLP
  tensors are bit-identical to the base, and the other expert groups moved.
  In `control`'s, the MLP tensors moved. (`check_frozen.py`, against E15's
  base dump.) If the MLP tensors of `frozen` moved, the arm is not what it
  claims and is not read.
* **V2 - every evaluation valid**: fifty episodes, client exit 0, `valid`
  true in the harness's own table.

---

## Predictions, and what falsifies each

**P1 - the collapse reproduces on this code.** `control` scores **at most 5
of 50**.

> Falsified by 6 or more. Then the collapse is not a property of FPO on this
> policy as the code stands today, `frozen` cannot be read as a remedy, and
> what changed becomes the question.

**P2 - freezing the MLP prevents it.** `frozen` scores **at least 20 of 50**.

> Read only if P1 holds. 20 is three standard deviations below the older
> null's mean. Holds: FPO's first iteration no longer destroys pi0.5 once the
> expert's MLP cannot move - a located cause with a remedy. Falsified, `frozen`
> at 5 or less: the rest of the expert, trained, destroys the policy too, and
> E22's "not sufficient alone" did not carry over to training. 6 to 19:
> partial, reported, no reading.

**Reported, not predicted:** `base`'s score, against the older null; each
arm's movement per module group (V1's output).

---

## Declared deviations allowed in advance

1. One restart of any run or evaluation that dies for a reason outside the
   experiment, recorded in `AMENDMENT.md`.
2. Cards may be reassigned if the planned ones are occupied.

---

## Reading order

V1, V2, then P1, then P2, then `base` and the movements.
