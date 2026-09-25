# E18 measurement protocol (pre-registered)

**Written 2026-09-25, before any E18 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E17 ran DPPO on `fpo-policy` and HalfCheetah for 409,600 steps on three seeds,
and it did not learn: `end − start` of −21.2, −8.4 and −28.5. Its write-up
named three configuration reasons and blamed none of them firmly, and one of
the three - DPPO being a fine-tuning method started from a random policy - was
presented as the leading one.

The day after, reading DPPO to fix one of the other two turned up a fourth
that E17 missed entirely, and got wrong in its own text. **DPPO never
maintained `fpo-policy`'s observation statistics.** That job lived in
`FPOAlgorithm.pre_learn` behind `isinstance(self.policy, FPOPolicy)`, and E17's
final checkpoint has `obs_stats_count` at 0.0 - mean zero, standard deviation
one, the identity - on an environment whose observation dimensions range in
standard deviation from 0.17 to 9.93. E17's document called those unchanged
statistics "frozen after warmup". They were never initialised.

PR #50 fixes that. **Does DPPO learn HalfCheetah once its policy's
observations are normalised?**

If it does, E17's failure was a defect in the platform, the "fine-tuning
method" explanation was at best secondary, and PlugRL has a second algorithm
that learns. If it does not, the defect was real and not sufficient, and the
fine-tuning hypothesis is what is left to test.

---

## Declared in advance: what was already known

Not part of the registered result.

1. **E17 is the null.** The same design with neither fix:

   | seed | start | best | end | end − start | best − start |
   | --- | --- | --- | --- | --- | --- |
   | 0 | −291.1 | −281.1 | −312.3 | −21.2 | +10.0 |
   | 1 | −313.1 | −281.1 | −321.5 | −8.4 | +32.0 |
   | 2 | −289.6 | −266.5 | −318.1 | −28.5 | +23.1 |

   The largest upward excursion of the ten-iteration mean anywhere in three
   hundred iterations of the null is **+32.0**. That number is what the
   threshold in P2 is set against.

2. **The gradient-accumulation fix (#49) is nearly inert here.** DPPO's
   optimizers are AdamW, which is invariant to a constant rescaling of every
   gradient except through its ε term, and E17 rescaled every step by exactly
   one quarter. Measured on seed 0 with that fix alone: identical return
   before the first update, about **4%** different after it. So although the
   code under test carries both fixes, E18 is in effect a test of the
   observation statistics, and the protocol says so rather than claiming to
   separate them.

3. **One-client training is deterministic on this machine.** The first
   iteration's return on seed 0 is identical to the fourth decimal between
   E17 and the #49 check run. So an E18 seed and the E17 seed of the same
   number differ only through the code.

---

## Held fixed

Everything E17 held fixed, unchanged: `fpo-policy` defaults on CPU;
`mujoco-v1` HalfCheetah-v5, one env per client; `--runner.replan-steps 1`;
server and client seeds 0, 1 and 2; DPPO's `cheetah` variant with the buffer
overridden to 4,096; `train_itrs` 100, so 409,600 steps; three runs at a time.

**Still from a random initialisation**, deliberately. Starting DPPO from a
trained policy is the other open question and it would change a second thing
at once. E18 changes only the code.

---

## What is measured

As E17: `rollout/reward`, with a run's value at a point being the mean of the
ten iterations ending there; `start` at the tenth iteration, `best` the
highest, `end` the last. `summarise.py` computes all three.

Plus the manipulation check below, read from the final checkpoint.

---

## Predictions, and what falsifies each

**P1 - the fix took effect in the run (manipulation check).** At the final
checkpoint of every seed, `obs_stats_count` is above zero and the largest
per-dimension `obs_stats_std` is more than ten times the smallest.

> Under the null that ratio is exactly 1.0 and the count is 0.0; FPO's own
> statistics in E16 give 59. Falsified if any seed fails either condition, in
> which case the run did not test what it was built to test and **nothing
> below is read**.

**P2 - DPPO learns.** On at least **2 of 3** seeds, `end − start` exceeds
**+200**.

> Set against the null, not chosen for being round: the largest upward
> excursion of the null's ten-iteration mean, at any point in any of its three
> seeds, was +32.0, and every null run ended below where it started. +200 is
> more than six times that excursion. Falsified otherwise, and then the
> observation statistics were a real defect that was not sufficient.

**P3 - wall clock.** Under 150 minutes for three concurrent seeds.

> E17 took 71 minutes for the same steps; the extra work per iteration is one
> pass over 4,096 observations. Falsified by exceeding it, reported as a cost
> result whatever the returns do.

**No prediction about FPO against DPPO.** The two do not share learning
rates, clipping, batch sizes or schedules, and nothing here should be read as
a comparison of the algorithms, exactly as in E17.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment - a
   suspension, a port collision, the machine running out of memory - recorded
   in `AMENDMENT.md`. A second such failure is reported, not worked around.
2. A seed that does not reach 409,600 steps is dropped. Two seeds is the
   minimum for any reading.

---

## Reading order

P1, then P3, then P2. P1 first because if the fix is not active in the run,
P2 is measuring E17 again.
