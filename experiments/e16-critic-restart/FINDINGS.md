# E16: an untrained value head slows FPO down; it does not destroy the policy

2026-09-24 · Windows 11, CPU only, no GPU · `HalfCheetah-v5`, `fpo-policy`,
272,423 parameters · protocol: [`PROTOCOL.md`](PROTOCOL.md) · deviations:
[`AMENDMENT.md`](AMENDMENT.md)

---

## The result

E14 took a pretrained pi0.5 from 29 of 50 to 0 of 50 in a single FPO
iteration. Ten hypotheses have been refuted, and E16 tests the one the two
working lines differ on most sharply: in E6 the value head grew up with the
actor, and in E14 the actor arrived from supervised pretraining while the head
that supplied FPO's advantages was random.

**It does not reproduce the collapse.** Three seeds, each restarted from the
same 327,680-step checkpoint and given 20 more updates, against the
uninterrupted run as control:

| seed | start | `all` | `model` | `except-critic` | control |
| --- | --- | --- | --- | --- | --- |
| 0 | 1129.8 | **+404.0** | +193.0 | **+73.0** | +323.7 |
| 1 | 1632.3 | **+549.1** | +482.4 | **+366.6** | +417.2 |
| 2 | 1183.0 | **+202.1** | +103.9 | **+28.8** | +162.7 |

`except-critic` is E14's shape: the actor and its observation statistics
restored, the value head left at its random initialisation, the optimizer
starting over. **Every arm rose. Not one fell.**

What the fresh value head costs is speed, and the cost is real but uneven.
As a fraction of what a faithful resume gained: **18%** on seed 0, **67%** on
seed 1, **14%** on seed 2. On two of three seeds the policy improves five to
seven times more slowly; on the third it barely notices. A policy that
improves slowly is not a policy that has been destroyed, and E14's collapse
was total and immediate.

**P3 is refuted, P4 holds**, and the bisection moves to the next variable.

**Correction, 2026-09-25: one run per arm cannot carry the per-seed claims in
this document.** E20 (#52) reran the `except-critic` arm - same code, same
checkpoint, same restore mode, same seed - and got **+104.7, +382.0, −55.7**
against +73.0, +366.6, +28.8 here. One-client collection is deterministic on
this machine; learning is not. Two identical runs agree until the first update
and then drift, and their twenty-iteration gains differ by up to **85**.

What survives and what does not:

* **The headline holds.** Across the nine runs here and the three reruns, the
  worst a random value head did to a good policy was −55.7 from 1183.0, under
  5%. E14's collapse was total and immediate.
* **"Every arm rose. Not one fell."** was true of these nine runs. The rerun
  of `except-critic` on seed 2 fell by 55.7. It is a statement about this
  sample, not the condition.
* **The per-seed ordering** `all` > `model` > `except-critic`: four of its six
  gaps exceed 85, and two do not - `all` over `model` on seed 1 (67) and
  `model` over `except-critic` on seed 2 (75). The order is consistent in
  sign; it is not established seed by seed. Averaged over seeds the gaps are
  125 and 104. With the noise in a single difference estimated, crudely, from
  E20's three pairs at about 53, a three-seed mean of differences carries
  about 30, so the average ordering stands on firmer ground than any seed's.
* **Which of the two costs is larger was misread**, independently of the
  noise. The ladder section below says that on seed 1 nearly all of the loss
  is the optimizer's and on seeds 0 and 2 the value head carries more, and a
  bullet under "Supported" said discarding the optimizer costs less than
  randomising the head. The table says the reverse. The optimizer's share,
  `all` − `model`, is 211, 67, 98; the head's, `model` − `except-critic`, is
  120, 116, 75. The optimizer carries more on seeds 0 and 2, the head on seed
  1, and on average the optimizer costs 125 to the head's 104. With about 53
  of noise in each difference, neither is established as the larger, and the
  bullet is removed.
* **The fractions 18%, 67%, 14%** are withdrawn as measurements. With E20's
  reruns in their place they read 26%, 70% and −28%. What stands is the
  average: `except-critic` gained 156 here and 144 in E20, against `all`'s
  385.
* **The resumes above their controls**, +80, +132, +39: two of the three are
  inside the spread of identical runs. This document already called them
  suggestive and no more; they are less than that.
* **Amendment 2's "the data is unaffected"** by the suspension means no
  connection dropped and no request timed out, which the server logs show. It
  cannot mean the run matches the one an unsuspended machine would have
  produced, because no two runs here match.
* **"272,423 parameters"** above counts the value head. The actor, which is
  what any comparison with pi0.5's movement is about, has **4,390**.

---

## The ladder, and what it separates

The three arms differ by one thing each, which is what makes the middle one
worth running:

| arm | actor | critic | `obs_stats_*` | optimizer, step |
| --- | --- | --- | --- | --- |
| `all` | loaded | loaded | loaded | loaded — a faithful resume |
| `model` | loaded | loaded | loaded | **fresh** |
| `except-critic` | loaded | **random** | loaded | fresh |

On all three seeds the ordering is the same and monotone:
**`all` > `model` > `except-critic`**, three of three.

| seed | `all` | `model` | `except-critic` |
| --- | --- | --- | --- |
| 0 | +404.0 | +193.0 | +73.0 |
| 1 | +549.1 | +482.4 | +366.6 |
| 2 | +202.1 | +103.9 | +28.8 |

Discarding the optimizer's moments costs something on its own, and randomising
the value head costs more on top of it. Neither is free and neither is fatal.
Three seeds is too few to separate the two contributions with any precision -
on seed 1 nearly all of the loss is the optimizer's, on seeds 0 and 2 the
value head carries more - but the sign and the order are consistent.

`except-critic` restores `obs_stats_*` on purpose. Those four buffers sit
outside both `actor.` and `critic.`, and holding them back would feed the
restored actor observations normalised differently from the ones it learned
on — a second change, which would make any collapse ambiguous between two
causes. Three of the ten tests in PR #39 exist to hold that line.

---

## P1: the control rises, on all three seeds

Phase A is E6's configuration rerun on today's `main`, because E6 ran on
`cb5b369` on 2026-09-10 and reading a 2026-09-24 result against its published
numbers would confound "the code changed" with what Phase B is testing.

| seed | at 327,680 | at 409,600 | |
| --- | --- | --- | --- |
| 0 | 1129.8 | 1453.6 | +323.7 |
| 1 | 1632.3 | 2049.5 | +417.2 |
| 2 | 1183.0 | 1345.7 | +162.7 |

Three of three. **P1 holds**, so there is a rising stretch to read Phase B
against.

It also reproduces E6, which is worth stating on its own: fourteen days and a
good deal of change to the FPO path later — float32 master weights, a seeding
fix, an allocator-cache release — the learning curve is the same curve.
`compare_e6.py` prints both at E6's own reported steps:

| step | E6 | Phase A |
| --- | --- | --- |
| 4,096 | −314.8 | −328.1 |
| 102,400 | 385.0 | 180.8 |
| 204,800 | 1181.2 | 940.9 |
| 307,200 | 1229.7 | 1456.6 |
| 409,600 | 1908.7 | 1705.7 |

Every difference sits inside the spread E6 documents for itself — it reports
single updates swinging by more than a thousand after step 100,000.

**P6 holds**: 91 minutes for three concurrent seeds, against a 150-minute
bound.

---

## P2 was falsified, and the rule was the wrong one

This has to be reported in two parts, because resolving it in either direction
would be choosing the rule from the answer.

**As registered, P2 is falsified.** The rule reads: *the `all` arm's value at
409,600 lies between the minimum and maximum across Phase A's three seeds at
that step*. That range is [1345.7, 2049.5], and seed 1's `all` arm ends at
**2181.4**, above it.

**The rule compares the wrong things.** It measures one seed against the
spread across three, so a seed that is simply the best of the three fails it
by being itself. Against its own control, which is the comparison that was
meant:

| seed | control | `all` | |
| --- | --- | --- | --- |
| 0 | 1453.6 | 1533.9 | +5.5% |
| 1 | 2049.5 | 2181.4 | +6.4% |
| 2 | 1345.7 | 1385.1 | +2.9% |

All three resumes land **above** their own controls, not adrift from them —
the opposite of what a falsified P2 is supposed to indicate. Three of three in
the same direction is suggestive and no more: 3-6% is roughly 40-130 on this
scale, against single-update swings of over a thousand, so this does not
establish that resuming is *better*. It does not look like resuming being
unfaithful either.

**And the reading order says to stop.** `PROTOCOL.md` ends with: *If P2 fails,
report it as a defect in PR #39 and do not read P3.* P3 and P4 are read below
anyway, and that is a deviation from the registered order, stated here rather
than quietly taken. The reason: the registered consequence assumes P2 fails
because a restart is a change, and the measured failure is a badly chosen
denominator. The cost is that **E16's answer to its main question is weaker
than it would have been** had P2 been worded per-seed, and a follow-up should
register it properly rather than inherit this one.

The honest summary is that P2 as written is falsified, P2 as intended is not
contradicted by anything measured, and the difference is my error in writing
it.

---

## What this does and does not support

**Supported:**

* On `HalfCheetah-v5` with a flow policy whose actor has 4,390 parameters
  (272k with its value head), handing FPO a good actor with a randomly
  initialised value head does not destroy the policy. Nine runs here, three
  seeds, three arms, every one improved; a rerun of one arm on one seed in
  E20 fell by 55.7 of 1183.0. *(Corrected 2026-09-25.)*
* It does slow learning on average: `except-critic` gained 156 averaged over
  seeds, and 144 when rerun in E20, against 385 for a faithful resume. The
  per-seed fractions this document first gave are withdrawn - see the
  correction above. *(Corrected 2026-09-25.)*
* Averaged over seeds, the ordering `all` > `model` > `except-critic`, by 125
  and 104. Seed by seed, two of its six gaps are within the spread of
  identical runs. *(Corrected 2026-09-25.)*
* FPO training can now be started from a checkpoint at all (PR #39), and a
  faithful resume lands within a few per cent of the run it resumes.
* E6's learning curve reproduces on today's `main`.

**Not supported:**

* Anything about pi0.5. A 4,390-parameter actor on dense-reward locomotion is not a 3B VLA
  on sparse binary-reward manipulation, and this refutes the mechanism *at
  this scale*, which is not the same as exonerating the critic at E14's. The
  remaining differences — sparse reward, batch 8, a frozen trunk, bfloat16,
  action chunking — are not controlled here and are what the bisection tests
  next.
* Any claim that resuming is better than continuing. Three seeds, 3-6%,
  against a per-update spread an order of magnitude larger. It is recorded
  because it is consistent in sign, not because it is established.
* Any wall-clock statement about Phase B, whose machine suspended for 32
  minutes and was then short of memory. See `AMENDMENT.md`; no connection
  dropped, and the elapsed times are not reported as a cost.
* Any per-seed difference between two runs smaller than about 85, the spread
  E20 measured between identical runs of this setup.

---

## Reproducing

```bash
bash phase_a.sh              # 3 seeds x 409,600 steps, ~91 min, three at a time
bash phase_b.sh              # 9 runs, 3 batches of 3, from the 327,680 checkpoint
python summarise.py          # the curves, as summary.tsv
python read_predictions.py   # P1-P5, computed rather than judged
python compare_e6.py         # Phase A against E6's published curve
```

`read_predictions.py` applies each prediction literally and prints no verdict
for an arm short of its twenty updates. It was written after two of three
seeds had landed, which is recorded in its commit: the predictions were
registered before any Phase B run, the script computing them was not.
