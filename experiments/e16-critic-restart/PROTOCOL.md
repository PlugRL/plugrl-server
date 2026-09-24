# E16 measurement protocol (pre-registered)

**Written 2026-09-24, before any run it describes.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

Two FPO results in this repository point opposite ways.

**E6** trained a 272,423-parameter flow policy on `HalfCheetah-v5` for 500,000
steps on a laptop CPU, three seeds. Episode return went from −315 ± 26 to
1928 ± 224 and **did not fall back**: each seed's final value is within a few
per cent of its own maximum.

**E14** trained a 3B pi0.5 on LIBERO for nine iterations. The policy went from
29 of 50 to **0 of 50 in a single iteration** and never recovered. Eleven
interventions have not located the cause, and five candidate explanations have
been refuted and written up.

So FPO, in this codebase, both works and destroys a policy. The difference is
not "small model against large" alone, because that difference is a bundle.
E16 starts unpacking it, and starts with the variable that the two runs differ
on most sharply and that nothing has tested:

**In E6 the value head grew up with the actor.** Both started random, and by
the time the actor was any good the critic had been fitted to it for dozens of
updates.

**In E14 the actor arrived pretrained and the value head was new.** pi0.5's
weights come from supervised pretraining, which has no value function in it;
the head that supplies FPO's advantages began at random initialisation. E14
measured `losses/value_loss` at 0.880 on the first iteration and 9.3e-05 by
the ninth — the critic could not predict returns at all at the moment it
weighted the first policy update.

The question:

**Does handing FPO a good actor and an untrained value head reproduce E14's
collapse, on a model small enough to debug?**

### Why this is worth a run rather than an argument

It is cheap — a laptop CPU, no GPU — and both endpoints already exist, so a
reproduction would move the pi0.5 problem onto a model that fits in memory,
runs in minutes, and can be instrumented freely. A refutation removes a
candidate and sends the bisection to the next variable.

### What this cannot settle

A 272k MLP on dense-reward locomotion is not a 3B VLA on sparse binary-reward
manipulation, and the remaining differences (batch 8, frozen trunk, bfloat16,
sparse reward, action chunking) are not controlled here. A collapse here
reproduces E14's **shape**; it does not prove the same cause operates at
scale. A non-collapse here does not exonerate the critic at scale either.

---

## Declared in advance: what was already known

Not part of the registered result:

1. **E6's published numbers**, above. They are a reference, not this
   experiment's control — `main` has moved a long way from E6's `cb5b369`,
   so reading a 2026-09-24 run against a 2026-09-10 number would confound
   "the code changed" with "the run was restarted". **Phase A reproduces the
   control on today's code**, and that is what Phase B is read against.
2. **Critic warmup was already tested on pi0.5 and refuted.** One warmup
   iteration with the actor frozen, verified frozen by an evaluation at 31 of
   50, then one real update returning 0 of 50. One iteration may simply be
   too few; this experiment tests the mechanism, not that fix.
3. **`value_loss_coeff` is inert** on the pi0.5 path: `freeze_vlm` sets
   `requires_grad = False` across the trunk, so the actor's and critic's
   parameters are disjoint and Adam's per-parameter normalisation cancels the
   scaling. Whatever the critic does to the actor, it does not do it through
   that coefficient.
4. **Nothing could load weights into an FPO training run** until
   `feat/fpo-resume-from-checkpoint` (PR #39), merged or not at the time of
   reading. `--algo.restore` and its three modes are that change; E16 is the
   first use of it.

---

## Held fixed

E6's configuration exactly, so Phase A is a reproduction and not a new setup:

| | |
| --- | --- |
| Policy | `fpo-policy`, defaults, `--policy.device cpu` |
| Algorithm | `fpo`, defaults except `--algo.buffer-size 4096` |
| Environment | `mujoco-v1`, `HalfCheetah-v5`, 1 env per client |
| Client | `--runner.replan-steps 1`, `--runner.seed $SEED` |
| Server seed | `--seed $SEED`, seeds 0, 1, 2 |
| Save interval | 20 updates → checkpoints at steps 81,920 … 409,600 |
| Concurrency | three runs at a time, as E6 ran |

`buffer_size=4096` gives one update per 4,096 environment steps, so 409,600
steps is 100 updates.

---

## Design

### Phase A — the control, and the checkpoints

Three seeds, from scratch, to **409,600 steps**. This is E6 rerun on today's
`main` plus PR #39, stopped at 100 updates instead of 122.

It produces two things: the uninterrupted stretch from 327,680 to 409,600
steps, which is the control, and the checkpoint at **327,680 steps** (update
80), which is what Phase B starts from.

327,680 is chosen because E6's published curve rises across that stretch on
**all three** seeds (1789.8 → 2210.5, 1049.1 → 2033.9, 850.2 → 1481.6 over
307,200 → 409,600). A stretch where the control is flat or falling would make
any Phase B reading unreadable; P1 below checks that it rose this time too,
before anything else is read.

### Phase B — three arms, one variable each

From each seed's 327,680-step checkpoint, **81,920 further steps** (20
updates), under three restore modes:

| arm | actor | critic | `obs_stats_*` | optimizer, step | what it isolates |
| --- | --- | --- | --- | --- | --- |
| `all` | loaded | loaded | loaded | loaded | nothing — a faithful resume |
| `model` | loaded | loaded | loaded | **fresh** | the optimizer's moments |
| `except-critic` | loaded | **random** | loaded | fresh | **the untrained value head** |

`except-critic` restores `obs_stats_*` deliberately. Those four buffers sit
outside both `actor.` and `critic.`; holding them back would feed the restored
actor differently normalised observations, which is a second change and would
make any collapse ambiguous between two causes.

Nine runs, in three batches of three. Each batch holds one run of each arm, so
CPU contention is identical across arms and cannot favour one of them.

### What is measured

`rollout/reward` from each run's tensorboard, the same scalar E6 reported,
extracted by `summarise.py`. A run's **value at a step** is the mean over its
last ten updates ending at that step, as E6 defined its endpoint; single
updates in E6 swung by over a thousand, so no single update is read as a
result.

---

## Predictions, and what falsifies each

**P1 — the control rises.** In Phase A, the value at 409,600 exceeds the value
at 327,680 on at least **2 of 3** seeds.

> Falsified if it does not. Then there is no rising stretch to compare
> against, Phase B's main question cannot be read here, and the honest report
> is that the experiment is void for its purpose — not a reading of Phase B
> against a flat control. P2 would still be readable.

**P2 — resuming is faithful.** The `all` arm's value at 409,600 lies between
the minimum and maximum across Phase A's three seeds at that step.

> Falsified if it falls outside. That would mean `--algo.restore all` does not
> reproduce an uninterrupted run, and the other two arms could not then be
> read as isolating anything, because restarting would itself be a change.
> This is a check on PR #39, and it is placed before the main question on
> purpose.

**P3 — an untrained value head reproduces the collapse.** On at least **2 of
3** seeds, the `except-critic` arm's value at 409,600 is **below its own
starting value** (that seed's Phase A value at 327,680), while the `all` arm's
is not.

> This is the result the experiment exists for. If it holds, E14's shape is
> reproduced at 272k parameters on a CPU, and the pi0.5 problem becomes
> debuggable at a scale where every tensor can be looked at.

**P4 — or it is refuted.** If `except-critic` ends above its starting value on
at least 2 of 3 seeds, the untrained-value-head mechanism does not by itself
produce E14's collapse at this scale, and the bisection moves to the next
variable: sparse binary reward, then batch size 8, then the frozen trunk.

**P5 — which of the two fresh things matters.** `model` differs from
`except-critic` only in that its value head is also restored. If `model`
collapses too, the cause is the discarded optimizer state and not the critic.
If `model` holds and `except-critic` falls, the critic is the difference.

> P5 has no independent falsification: it is read only when P3 or P4 has
> already been decided, and it reports which of the two changes carries the
> effect. Stated in advance so that the answer cannot be chosen afterwards.

**P6 — wall clock.** Phase A completes within 150 minutes per batch of three.
E6 measured 98-103 minutes for 500,000 steps with three runs sharing 16 cores;
409,600 steps is 82% of that, so ~85 minutes is expected and 150 is a generous
bound that catches a hang rather than a slowdown.

> Falsified by exceeding it, which would be reported as a cost result
> regardless of what the returns do.

---

## Declared deviations allowed in advance

1. **One restart** of any run that dies for a reason outside the experiment
   (a machine sleeping, a port collision, an unrelated process). Recorded in
   `AMENDMENT.md` with the reason. A second such failure is reported, not
   worked around.
2. If a Phase A seed fails to reach 409,600 steps, its Phase B arms are
   dropped and the seed count for P1, P3 and P4 falls to what remains. Two
   seeds is the minimum for any reading; at one seed the experiment reports
   only that it could not be run.

---

## Reading order

P1, then P2, then P3/P4, then P5, then P6. If P1 fails, stop at P2 and say so.
If P2 fails, report it as a defect in PR #39 and do not read P3.
