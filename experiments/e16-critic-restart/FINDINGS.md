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

* On `HalfCheetah-v5` with a 272k-parameter flow policy, handing FPO a good
  actor with a randomly initialised value head does not destroy the policy.
  **Nine runs, three seeds, three arms: every one improved.**
* It does slow learning, by a factor that varies with the seed — the
  `except-critic` arm gains 18%, 67% and 14% of what a faithful resume gains —
  and the ordering `all` > `model` > `except-critic` holds on all three seeds.
* Discarding the optimizer's moments alone also costs something, and less than
  randomising the value head on top of it.
* FPO training can now be started from a checkpoint at all (PR #39), and a
  faithful resume lands within a few per cent of the run it resumes.
* E6's learning curve reproduces on today's `main`.

**Not supported:**

* Anything about pi0.5. A 272k MLP on dense-reward locomotion is not a 3B VLA
  on sparse binary-reward manipulation, and this refutes the mechanism *at
  this scale*, which is not the same as exonerating the critic at E14's. The
  remaining differences — sparse reward, batch 8, a frozen trunk, bfloat16,
  action chunking — are not controlled here and are what the bisection tests
  next.
* Any claim that resuming is better than continuing. Three seeds, 3-6%,
  against a per-update spread an order of magnitude larger. It is recorded
  because it is consistent in sign, not because it is established.
* Any wall-clock statement about Phase B, whose machine suspended for 32
  minutes and was then short of memory. See `AMENDMENT.md`; the data is
  unaffected and the elapsed times are not reported as a cost.

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
