# E15: is the collapse a function of the trust region?

**Written 2026-09-23 21:45, while the runs are in flight and before any of
their evaluations has returned.**

This is **not a pre-registration of the experiment**. The five runs were
already launched when this was written, and two of their results were already
known. What is registered here is the **rule for reading the other three**,
committed before they land, because a curve is easy to narrate either way
afterwards.

## What prompted it

E14 put a pi0.5 policy through FPO and it went from 29 of 50 to 0 after one
iteration, and stayed at 0. Three single-knob variants were run, each of which
reduces how far one iteration moves the policy:

| knob | change | result |
|---|---|---|
| `num_updates_per_batch` | 4 → 1, so 512 gradient steps instead of 2048 | **0 of 50** |
| `learning_rate` | 1e-5 → 1e-6 | **0 of 50** |
| **`clipping_epsilon`** | **0.05 → 0.01** | **8 of 50** |

Taking fewer steps did nothing. Taking smaller steps did nothing. Moving the
boundary did something. That points at the policy walking to the edge of its
trust region within a single iteration and the edge being what determines the
damage - not the number of steps taken to reach it.

All five runs carry the buffer-level advantage normalisation from
`fix/advantage-normalisation-scope`. That change was made for a different
hypothesis, which its own two-iteration run refuted at 0 of 50; it is in the
code these arms run on and is named here so the arms are not read as testing
it.

## The three runs in flight

`clipping_epsilon` at **0.02**, **0.005** and **0.002**. One iteration each,
everything else identical to E14: buffer 4096, batch 8, four updates per
batch, four samples per action, learning rate 1e-5, `libero_10` task 8, ten
clients, seed 7. Each is evaluated over 50 episodes, states 0-49 in order.

With the two already measured, that gives five points: 0.05, 0.02, 0.01,
0.005, 0.002.

## How the result will be read

**If the five points are monotone** - success rate rising as the trust region
tightens - the mechanism is supported and 8 of 50 was not noise.

**If they are not monotone**, or if 0.005 and 0.002 come back at or near zero
while 0.01 stands alone at 8, then 8 of 50 is a fluctuation, the mechanism is
not supported, and this document will say so.

Either way the headline number is bounded by what it is: the baseline scores
29 of 50, so an arm at 8 is a policy that has been damaged rather than
preserved, and no arm short of the high twenties is a policy that survived.

## What must be re-measured before any of this is reported

**The evaluation is not reproducible when it loads a checkpoint.** Measured
while these arms were running:

| path | runs | results |
|---|---|---|
| no checkpoint | 3 | **29, 29, 29** |
| checkpoint, same file, bit-identical policy | 2 | **35, 28** |

An in-process comparison shows that loading changes only the critic's four
weight matrices, leaves every actor tensor bit-identical, and does not touch
the CPU RNG state or any module's training mode. So the spread is not the
weights, and it is not yet explained.

Every arm above is evaluated through the checkpoint path. A single arm's
number therefore carries an unquantified spread, and **`clipping_epsilon`
0.01 must be evaluated a second time** before 8 of 50 is reported as anything.
The curve's shape across five points is more robust to this than any single
point is, which is why the curve is the claim and not the 8.
