# E19 measurement protocol (pre-registered)

**Written 2026-09-25, before any E19 run, while E18 was still running.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E17 put DPPO in front of a randomly initialised policy and it did not learn.
Its write-up said the experiment DPPO deserves is the one it was built for:
**fine-tune a policy that is already good.** DPPO's paper pretrains by
behaviour cloning and then does RL; nothing in this project had put it in
that setting, because DPPO could not start from a checkpoint until #51.

E16 left three good policies at step 327,680, and ran FPO on them from
exactly the configuration E19 needs: the actor and its observation statistics
restored, the value head at a fresh random initialisation, the optimizer new.
That was E16's `except-critic` arm.

**E19 is that arm with DPPO instead of FPO.** Same three checkpoints, same
restore mode, same twenty iterations. Does DPPO hold a good policy, and what
does it do with it?

### What this cannot settle

**Which algorithm is better.** DPPO and FPO do not share learning rates,
clipping, batch sizes or schedules, and nothing here is a comparison of them.
E16's numbers are printed beside E19's as a reference, with no verdict.

And one iteration of each algorithm is a different amount of optimisation, so
"twenty iterations" is matched in environment steps, not in gradient steps.

---

## Declared in advance: what was already known

1. **E16's `except-critic` arm, FPO, twenty iterations from these checkpoints:**
   +73.0, +366.6, +28.8 on seeds 0, 1, 2, measured from Phase A's value at
   the checkpoint. Every one of E16's nine restarted runs rose.

2. **A random policy's first-iteration return**, from E17 and E18, which agree
   to the decimal on it: −356.0, −356.8, −290.2 on seeds 0, 1, 2.

3. **The sampling differs, and that changes what "start" can mean.** FPO
   collects with deterministic denoising. DPPO collects with
   `sampling_noise_level` 0.1, because its log-probabilities need a density.
   Phase A's value at the checkpoint was measured under FPO's sampling; E19's
   returns are measured under DPPO's. Reading E19's end against Phase A's
   value would count whatever the noise costs as something DPPO did to the
   policy. **So E19's start is its own first iteration**, whose collection
   happens before any DPPO update and therefore measures the restored policy
   under DPPO's sampling, untouched. Phase A's value is printed for reference
   and not used for any verdict.

   What the noise costs is **not known in advance**, and that is why P1(b)
   below is a real test rather than a formality.

---

## Held fixed

`fpo-policy` on CPU; `mujoco-v1` HalfCheetah-v5, one env per client;
`--runner.replan-steps 1`; server and client seeds matched to the checkpoint's
seed; DPPO's `cheetah` variant with the buffer at 4,096; **`train_itrs` 20**,
so 81,920 steps; `--algo.restore except-critic` from the E16 Phase A checkpoint
nearest step 327,680 for that seed; three runs at a time.

The checkpoint is found, not named: Phase A wrote seed 0's at 327680 and
seeds 1 and 2's at 327681, which is the defect E16's amendment 1 records.

---

## What is measured

`rollout/reward` per iteration. **`start`** is the first iteration's value;
**`end`** is the mean of the last ten (iterations 11-20). A single iteration
on HalfCheetah moves by 100-150 on its own - E17's seed 0 went from −356 to
−232 across one update - which is why `start` is compared against thresholds
far larger than that and never against another single iteration.

---

## Predictions, and what falsifies each

**P1 - the restore took effect.** Two parts, reported separately, because
they fail for different reasons.

> **(a) Mechanically:** every server log records the restore with
> `restore=except-critic` and holding back twelve critic tensors. If not, the
> run did not load a policy and nothing below is read.
>
> **(b) Behaviourally:** on every seed, the first iteration's return exceeds
> that seed's random-policy first iteration (known item 2) by more than
> **500**. The restored policies scored 1130 to 1632 under FPO's sampling,
> 1400 to 1900 above random, so 500 leaves room for the noise to cost a great
> deal. If (a) holds and (b) fails, the policy loaded and DPPO's sampling
> noise makes it act close to randomly - a result about the setting, reported
> as such, with P2 read under that caveat.

**P2 - DPPO does not collapse a good policy.** On every seed, `end − start`
exceeds **−500**.

> This is E14's failure mode transplanted: a trained policy handed to an RL
> algorithm and destroyed. Collapse to random-policy level from these
> starting points would be a drop of well over 1,000, and the single-iteration
> noise in `start` is 100-150, so −500 separates the two with margin on both
> sides. Falsified by any seed below it.

**P3 - wall clock.** Under 45 minutes for three concurrent seeds.

> E17 and E18 ran a hundred iterations in about 70 minutes, so twenty should
> take fifteen plus startup. Falsified by exceeding it.

**Not a prediction: whether DPPO improves the policy.** `end > start` on at
least two of three seeds would happen about half the time under a null of no
systematic change, which makes it useless as a registered test. `end − start`
is reported per seed beside E16's FPO arm, without a verdict. E16 and E17
both registered thresholds without checking what they took under the null;
this one was checked and dropped.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`. A second is reported, not worked around.
2. A seed that does not finish twenty iterations is dropped. Two is the
   minimum for any reading.

---

## Reading order

P1(a), P1(b), P3, P2. The improvement figures last, as description.
