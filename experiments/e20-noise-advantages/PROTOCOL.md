# E20 measurement protocol (pre-registered)

**Written 2026-09-25, before any E20 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E14's pi0.5 lost 29 of 50 in one FPO iteration. The evidence since points at
the *direction* of the update rather than its size: E15 showed under 1% of
relative movement is enough, and four independent ways of moving less - fewer
steps, a smaller learning rate, a tighter clip, an explicit drift stop - all
scored zero, while a learning rate of zero scored 37.

E16 then ruled out the obvious candidate on its own. A good policy handed a
**randomly initialised value head** on HalfCheetah did not collapse; it
improved more slowly. But HalfCheetah's reward is dense, and E14's is not.

Put the two together and there is a mechanism that fits everything:

> With a random critic **and no reward signal**, an advantage
> `r + γV(s') − V(s)` is the critic's noise and nothing else. FPO normalises
> advantages to unit variance, so that noise is applied at full strength, in
> directions unrelated to the task. A precise pretrained policy has very
> little room - E15's under 1% - and a random direction spends it.

LIBERO's reward is binary and sparse, and in four thousand transitions most
of it is zero. HalfCheetah's is never zero. E16 changed the critic and kept
the reward; **E20 keeps E16's random critic and removes the reward**, which
`--algo.reward-scaling 0` does without new code: FPO stores
`reward * reward_scaling`.

**Does a good policy collapse when its advantages are a random critic's
noise?** If it does, E14's shape is reproduced at 272k parameters on a CPU by
a mechanism that can be named. If it does not, the mechanism is wrong.

### What this cannot settle

Zero reward is the limit of sparse reward, not sparse reward. LIBERO's is
zero almost everywhere and one at success; this is zero everywhere. A collapse
here shows what noise advantages do; it does not show that LIBERO's
advantages are noise to the same degree. That is the next thing to measure,
and it is measurable from E14's own logs.

---

## Declared in advance: what was already known

1. **E16's `except-critic` arm** - the same checkpoints, restore mode, code and
   twenty iterations, with the reward on: +73.0, +366.6, +28.8 on seeds 0, 1,
   2, against starts of 1129.8, 1632.3, 1183.0.
2. **One-client training is deterministic on this machine.** The first
   iteration's return agreed to the fourth decimal between E17 and a later run
   of the same seed, and again between E17 and E18.
3. **The code is E16's.** This branch is cut from `exp/e16-critic-restart`,
   so the only difference between E20's reward-on arm and E16's is the day it
   ran.

---

## Design

Two arms, each from Phase A's checkpoint nearest step 327,680 for its seed,
with `--algo.restore except-critic`, twenty iterations of 4,096 steps:

| arm | `reward_scaling` | what it is |
| --- | --- | --- |
| `reward` | 10, the default | E16's `except-critic` arm, rerun |
| `zero` | **0** | the same, with every stored reward zero |

Six runs in two batches of three; a batch is one arm's three seeds, so the two
arms meet the same contention.

`start` and `end` are E16's: the seed's Phase A value at the checkpoint, and
the mean of the arm's last ten iterations. FPO collects with deterministic
denoising, so unlike E19 the Phase A value is measured under the same sampling
as the arms and is a fair start.

---

## Predictions, and what falsifies each

**P1 - the reward-on arm reproduces E16.** On every seed, `end − start` for
the `reward` arm is within **1.0** of E16's (known item 1).

> Determinism says it should match to far better than that. Falsified by any
> seed off by more than 1.0, and then the two arms cannot be read as
> differing in one variable, because something other than the reward changed
> between E16 and today. P2 is still reported and not read as a result.

**P2 - noise advantages collapse a good policy.** On at least **2 of 3**
seeds, the `zero` arm's `end − start` is below **−500**.

> What a collapse looks like here: back towards a random policy's −300 or so,
> a drop of 1,400 to 1,900 from these starts. The honest statement about the
> other side is that the null for this condition has not been measured -
> nobody has run FPO with zero reward before - so a `zero` arm that falls but
> by less than 500 is reported as falling, without a verdict either way.
> E16 and E17 registered thresholds without checking the null; this one could
> not be checked, and says so rather than pretending otherwise.

**P3 - wall clock.** Under 60 minutes for all six runs.

> E16's Phase B batches took about fifteen minutes each when the machine was
> not suspending. Falsified by exceeding it.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.
2. A seed that does not finish is dropped; two is the minimum.

---

## Reading order

P1, then P3, then P2. If P1 fails, P2 is reported but not read.
