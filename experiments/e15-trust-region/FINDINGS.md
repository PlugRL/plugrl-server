# E15: tightening the trust region does not preserve the policy

2026-09-24 · qz103, 8× RTX 3090 · `libero_10` task 8, 50 episodes per
evaluation, states 0-49 in order · protocol: [`PROTOCOL.md`](PROTOCOL.md)

---

## The result

E14's pi0.5 went from 29 of 50 to 0 of 50 in one FPO iteration. Of three
single-knob variants, only `clipping_epsilon` moved the number at all — 0.05
to 0.01 gave 8 of 50 where fewer gradient steps and a smaller learning rate
each gave 0. That suggested the policy walks to the edge of its trust region
within one iteration and the edge is what sets the damage.

`PROTOCOL.md` registered the rule for reading the three further points before
they landed: **monotone supports the mechanism; not monotone means 8 of 50 was
a fluctuation.** Five points, one iteration each, everything else identical to
E14:

| `clipping_epsilon` | successes of 50 | Wilson 95% |
| --- | --- | --- |
| 0.05 (E14's own) | 0 | 0.000 – 0.071 |
| 0.02 | 0 | 0.000 – 0.071 |
| **0.01** | **8** (repeat: 7) | 0.083 – 0.285 |
| 0.005 | 2 (repeat: 2) | 0.011 – 0.135 |
| 0.002 | 0 | 0.000 – 0.071 |
| *baseline, no training* | *29* | *0.442 – 0.706* |

**Not monotone.** By the registered rule the mechanism is not supported, and
that is the finding.

One correction to the rule's own wording is owed. It anticipated that a
non-monotone curve would mean "8 of 50 is a fluctuation", and that is not what
the repeats show. 0.01 was evaluated twice and returned 8 and 7; 0.005 twice
and returned 2 and 2. The 0.01 interval (0.083 – 0.285) does not overlap the
intervals at 0.05, 0.02 or 0.002. **The bump at 0.01 is real and it repeats.**
What is refuted is not the measurement but the mechanism the measurement was
supposed to support: success does not rise as the trust region tightens, and
tightening past 0.01 makes things worse again, which no account of "the edge
sets the damage" predicts.

And the bound the protocol set on itself holds in every case: the baseline is
29 of 50, the best arm is 8, and the two do not overlap. **No setting of
`clipping_epsilon` produced a policy that survived.** The best of them is a
policy that has been damaged less.

---

## Why the trust region could not have been the knob

Measured after the arms had run, over the checkpoints themselves: how far each
arm moved the action expert's 201 tensors, relative, from the policy it
started from.

| arm | action expert moved | score |
| --- | --- | --- |
| `lr0-control` (learning rate 0) | **0** | 37 of 50 |
| learning rate 1e-6 | 0.0066 | 0 of 50 |
| `clipping_epsilon` 0.002 | 0.0087 | 0 of 50 |
| `num_updates_per_batch` 1 | 0.0099 | 0 of 50 |
| E14's own first iteration | 0.0151 | 0 of 50 |

Two things fall out of that table.

**A policy that moves less is not damaged less.** The learning-rate arm
travelled 0.0066 where E14 travelled 0.0151 — well under half the distance —
and returned the same 0 of 50. Every arm between 0.66% and 1.5% scores zero.
There is no dose-response here to find a safe dose inside.

**Under one per cent of relative movement in the action expert is enough to
take a pi0.5 from 29 of 50 to 0 of 50.** That is the number worth carrying
out of E15. It is not a statement about FPO's trust region; it is a statement
about how little room this policy has.

The control anchors both: with the learning rate at zero the actor's 821
tensors are bit-identical to the pretrained ones — every actor group reads
exactly 0 — and it scores 37 of 50. So the movement above is FPO's doing, not
the harness's, and the 1.41 that the critic shows in every row is the distance
between two random initialisations rather than anything training did.

`clipping_epsilon` 0.01, the one arm that scored above 2, was **not** measured
this way. Whether the arm that survives best also moves least is the obvious
next question and this experiment does not answer it.

---

## The measurement's own defect, and what was done about it

**Evaluations that load a checkpoint are not reproducible.** Measured while
these arms were running:

| path | runs | results |
| --- | --- | --- |
| no checkpoint | 3 | 29, 29, 29 |
| checkpoint, same file, bit-identical policy | 3 | 35, 28, 29 |

An in-process comparison shows that loading changes only the critic's four
weight matrices, leaves every actor tensor bit-identical, and touches neither
the CPU RNG state nor any module's training mode. **The spread is not the
weights and it is not explained.**

Every arm above is evaluated through the checkpoint path, so every single
number in this document carries an unquantified spread of that size. The
protocol required `clipping_epsilon` 0.01 to be evaluated twice before 8 of 50
was reported as anything; it was, and returned 7. 0.005 was repeated too, and
returned 2 both times.

That is mitigation, not a fix. A spread of roughly ±4 on a no-checkpoint
baseline of 29 is large enough to matter for any arm in the twenties, and this
experiment has no arm in the twenties — which is the only reason its
conclusion survives the defect. An arm at 8 against a baseline at 29 is not a
call the spread can flip.

---

## The control this experiment did not have

Nine hypotheses about E14's collapse have now been refuted, five of them at
hours each on the cluster, one knob at a time — and **none of them could have
found a defect, because defects do not live in hyperparameters.** The sweep in
this document is the fourth of those five.

[`results/toy/FINDINGS.md`](results/toy/FINDINGS.md) is the control that was
missing: a bandit on a two-layer flow policy, on a CPU, about a minute per
run, where the right answer is known. It establishes that FPO learns, that
**FPO does not hold what it learns** — four of ten runs end more than twice
below their best — and that a diverging value loss tracks the damage at rank
correlation 0.988. It also withdraws a dtype claim this line of work had made
on one seed.

Read that document before this one if you only read one. This one measures a
knob; that one measures the algorithm.

---

## What this does and does not support

**Supported:**

* Success on `libero_10` task 8 does not rise monotonically as FPO's trust
  region tightens. Five points, 0.05 to 0.002, with the two interior points
  repeated.
* The 8 of 50 at `clipping_epsilon` 0.01 is reproducible (8, then 7) and sits
  above its neighbours on both sides with non-overlapping intervals.
* No `clipping_epsilon` in that range preserves the policy: baseline 29 of 50,
  best arm 8 of 50, intervals disjoint.
* Relative movement under 1% in pi0.5's action expert is sufficient to take it
  from 29 of 50 to 0 of 50, and roughly halving that movement does not reduce
  the damage.
* With the learning rate at 0, every actor tensor is bit-identical to the
  pretrained policy and the evaluation returns 37 of 50.

**Not supported:**

* Anything about why 0.01 is better than both 0.02 and 0.005. The bump is
  measured, not explained.
* Anything about `clipping_epsilon` outside [0.002, 0.05], on another task, on
  another seed, or over more than one iteration.
* Any number here to better than the spread the checkpoint-loading defect
  introduces, which is unquantified and at least ±4 of 50 on the baseline.
* That the weight-distance table generalises past these four arms. It does not
  include the arm that scored best.

---

## Reproducing

The harness as run, the per-arm evaluation rows and the diagnostic scripts are
all in `results/`. `results/eval.tsv` carries every evaluation in this
document, including the ones from E14 that the arms are read against.
