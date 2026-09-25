# E21 measurement protocol (pre-registered)

**Written 2026-09-25, before any E21 checkpoint was built or evaluated.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

One FPO iteration takes pi0.5 on LIBERO-10 task 8 from about 30 of 50 to 0 of
50 (E14). E15 measured how far that iteration moves the policy: **1.51%** of
the action expert's norm, relative, and 0.66% to 0.99% for the three arms that
moved less and still scored zero. E20 then ran the mechanism that looked most
likely - advantages that are a random critic's noise, because the reward is
zero - on HalfCheetah, and the good policies there **did not** collapse,
although their actors moved 11% to 22%, ten times as far.

So either of two things is true, and they lead to different fixes:

* **A. pi0.5 has no room.** Any movement of about 1% destroys it, whatever
  its direction. Then nothing about FPO's update is wrong; the only fix is to
  move less than E15's smallest arm, or not at all in these tensors.
* **B. FPO's direction does the damage.** A movement of the same size in a
  random direction is harmless, and the update FPO chose is not. Then the fix
  is in whatever sets the direction.

**E21 separates the two by moving pi0.5 exactly as far as FPO did, tensor by
tensor and element by element, in directions FPO did not choose.** No
training is involved: each arm is a checkpoint built from the base policy and
FPO's own update, and evaluated with E14's harness unchanged.

### The random control, and why it is this one

`Δ` is FPO's first-iteration update: E14's first checkpoint minus the base,
per tensor, computed in float32 from the stored values, which is exact.

The random arms apply `θ + k · r ⊙ Δ`, with `r` a random ±1 per element. That
keeps **everything about FPO's update except its direction**: which tensors
move, which elements, and by how much each one moves. Only the sign pattern
is random. A Gaussian direction scaled to the same norm would change the
per-element profile as well, and would also be distorted on saving: 126 of
the 208 tensors FPO changes are stored in bfloat16, and a probe of `Δ`
(`probe_delta.py`, output in `results/probe-delta.txt`) shows its bf16
elements move by 0 representable steps (12%), 1 (20%), 2 (15%), and 10 or
more (19%). Flipping the sign of a move of whole steps stays on the grid
almost everywhere.

The mirror arms apply `θ − k · Δ`: FPO's exact update, backwards. They split
B in two:

* **B1, first order.** FPO's update points downhill on success. Its mirror
  points uphill or does nothing. This is what a sign or credit-assignment
  error in the objective would produce.
* **B2, second order.** FPO moves along directions in which the pretrained
  policy is sharp, and **either sign hurts**. This is what an update made of
  noise-weighted, demonstration-like gradients would produce: E20's known
  item 4 showed that E14's first update ran on advantages about 99.86% of
  which came from an untrained value head.

### What this cannot settle

* One task, fifty initial states, one evaluation per arm. The null is measured
  on the same task and states.
* The random control is random in sign, not isotropic. That is deliberate -
  it holds constant everything except direction - but it means "random" here
  is "random within FPO's own profile".
* If B holds, E21 does not say whether FPO's direction is harmful because of
  its **advantages** or because of its **objective** - the CFM-difference
  ratio - acting on a flow policy of this size. That is the next experiment,
  and which one depends on the answer to P2.

---

## Declared in advance: what was already known

1. **The null, an unperturbed actor**, from E15 correction 2: seven
   evaluations on this harness, task and states whose actor weights are
   bit-identical to the base scored 29, 29, 29, 35, 28, 29, 37 - **30.9 ± 3.6,
   range 28 to 37**. The spread lives in about eleven borderline initial
   states; a single forward is bit-identical from run to run.
2. **FPO-moved policies**, same harness: E14's first iteration 0; E15's arms
   at 0.66%, 0.87% and 0.99% scored 0, 0, 0; `max_policy_drift` 0.01 scored
   0 and 0; `clipping_epsilon` 0.01 scored 8 and 7; 0.005 scored 2 and 2.
3. **What `Δ` contains**, from `probe_delta.py` on E14's first checkpoint
   against the base, run before this file was written: 208 tensors changed -
   200 of the action expert's 201, plus the 8 tensors of `action_in_proj`,
   `action_out_proj`, `time_mlp_in` and `time_mlp_out`. 82 are float32 and
   126 bfloat16. No backbone tensor changed. The action expert's relative
   distance is 0.0151 (E15, `weight-distances.txt`).
4. **The thresholds were checked against the null** before being written
   down: an unperturbed actor has never scored below 28 in seven evaluations.

---

## Design

Each arm is a full checkpoint: the base's 821 tensors, with only the 208 that
FPO changed replaced. `r` is drawn per element from `torch.Generator` seeded
with the arm's seed, tensors in sorted key order, so arms with the same seed
share the same sign pattern and differ only in `k`. Every result is cast back
to the tensor's stored dtype, rounding to nearest.

| arm | tensors changed to | what it asks |
| --- | --- | --- |
| `flip-s1` | `θ + r₁ ⊙ Δ` | random direction, FPO's size |
| `flip-s2` | `θ + r₂ ⊙ Δ` | the same, a second direction |
| `flip-s3` | `θ + r₃ ⊙ Δ` | the same, a third direction |
| `neg` | `θ − Δ` | FPO's direction, backwards |
| `flip-s1-x3` | `θ + 3 r₁ ⊙ Δ` | `flip-s1`'s direction, three times as far |
| `flip-s1-x10` | `θ + 10 r₁ ⊙ Δ` | ten times |
| `flip-s1-x30` | `θ + 30 r₁ ⊙ Δ` | thirty times |
| `neg-x3` | `θ − 3 Δ` | FPO backwards, three times as far |

A ninth checkpoint, `fpo` = `θ + Δ`, is built as a check and **not
evaluated**: it must reproduce E14's first-iteration actor bit for bit, and
that actor is already known to score 0.

Evaluation is E14's harness, `e14/e14_eval.sh`, unchanged: task 8, fifty
episodes, initial states 0 to 49 in order, `runner.seed` 7, replanning every
5 steps, one env client. Two batches of four, each arm on its own pair of
cards, as E15's arm evaluations ran:

* batch 1: `flip-s1`, `flip-s2`, `neg`, `flip-s1-x3`
* batch 2: `flip-s3`, `flip-s1-x10`, `flip-s1-x30`, `neg-x3`

Batch 1 holds everything P1 and P2 need except `flip-s3`, so a failure in
batch 2 cannot leave the main question unanswered.

---

## Checks before any evaluation

`make_arms.py` builds all nine and refuses to exit 0 unless:

* **V1** - `fpo`'s 208 changed tensors are bit-identical to E14's first
  checkpoint.
* **V2** - in every arm, the 613 tensors outside the 208 are bit-identical to
  the base, read back from the saved file, with the same keys, shapes and
  dtypes.
* **V3** - every arm's realised relative distance in the action expert,
  E15's measure, is within **5%** of `k × 0.0151`, after rounding to the
  stored dtype.

If any check fails, nothing is evaluated; the construction is fixed and the
fix recorded in `AMENDMENT.md`.

After evaluation:

* **V4** - every evaluation is valid: fifty episodes, client exit 0, `valid`
  true in the harness's own table. An invalid arm is not read.

---

## Predictions, and what falsifies each

**P1 - random directions at FPO's size are harmless.** Each of `flip-s1`,
`flip-s2` and `flip-s3` scores **at least 20 of 50**.

> 20 is three standard deviations below the null's mean and eight below its
> lowest value. Under a normal approximation to the null, an unperturbed
> actor would score 19 or less about once in a thousand evaluations.
> Falsified by any of the three scoring 19 or less.
>
> Holds: B. The size of FPO's movement is not what destroys the policy.
> All three at 5 or less: A. pi0.5 has no room for 1.5% in any direction.
> Anything else: reported as partial damage, with no reading.

**P2 - the damage is second order.** `neg` scores **at most 10 of 50**.

> Read only if P1 holds. Holds: B2 - FPO moves along directions this policy
> is sharp in, either sign destroys, and a sign error in the objective is not
> the explanation. `neg` at 20 or more: B1 - the update points the wrong way
> specifically, and a sign or credit-assignment error becomes the lead
> suspect. 11 to 19: reported, no reading.
>
> Why this prediction and not the other: with advantages that are almost all
> an untrained value head's noise, FPO's update is a random combination of
> per-sample flow-matching gradients. Those gradients lie in the directions
> the demonstrations pinned down most tightly, where the loss curves most
> sharply, and moving along them either way costs the same to second order.

**P3 - the dose-response is reported, not predicted.** The scores of
`flip-s1` at 1, 3, 10 and 30 times, and of `neg` at 1 and 3 times, with the
smallest multiple at which the random direction scores 5 or less, if any.
There is no threshold to pass; this is the measurement that says how much
room pi0.5 has in a direction FPO did not choose.

---

## Declared deviations allowed in advance

1. One restart of any evaluation that dies for a reason outside the
   experiment - a server that never listens, a client crash - recorded in
   `AMENDMENT.md`.
2. An arm may move to another pair of cards if its own pair is occupied.

---

## Reading order

V1 to V3, then V4, then P1, then P2 if P1 holds, then P3.
