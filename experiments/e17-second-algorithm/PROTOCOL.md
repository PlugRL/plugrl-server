# E17 measurement protocol (pre-registered)

**Written 2026-09-24, before any DPPO training run it describes.** A twelve
iteration smoke run came first and is declared below rather than predicted.

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

Every learning result this project has is FPO. E6 trained a flow policy on
`HalfCheetah-v5`; E10, E11 and E14 put FPO in front of a pi0.5. When FPO
destroyed a policy in E14 there was nothing to compare it against, and eleven
interventions later there still is not: a defect in the FPO path and a
property of this setting look identical when only one algorithm has ever run.

`dppo` was in the CLI menu the whole time and had never executed. It reached
into an external package for two utility classes, and the module carrying its
algorithm class was never imported, so selecting it died with
`KeyError: 'Algorithm dppo is not registered.'` PR #40 fixes both.

**Does a second algorithm train the same policy on the same environment
through the same platform?**

### Why DPPO and not PPO

DPPO is the sibling approach to FPO's own problem. Both exist because a
diffusion or flow policy has no tractable action density: FPO replaces the
likelihood ratio with an exponentiated difference of flow-matching losses,
DPPO treats the denoising chain as an MDP and uses the per-step Gaussian
density instead. Two answers to one question is a comparison worth having;
PPO against a policy neither was designed for is not.

It can drive `fpo-policy` at all because `BasePolicyGradientFlowPolicy`
derives from the diffusion base class DPPO is typed against, and because
`sampling_noise_level` turns each deterministic denoising step into a Gaussian
transition that has a density. At `None` the log-probability is exactly zero
and DPPO would have no ratio to form. Every shipped `dppo` config sets it.

### What this cannot settle

**This is not a fair contest and no result here should be read as one.** The
two algorithms share the policy, the environment, the buffer size and the step
budget; they do not share learning rates, clipping, batch size, update counts
or schedules, because those do not correspond one-to-one between them. DPPO
runs on its shipped `cheetah` variant, which nobody has tuned for a 4,096-step
buffer.

So a difference in final return is **not** attributable to the algorithm. What
is attributable is whether each one learns at all, and whether each one keeps
what it learned.

---

## Declared in advance: what was already known

Not part of the registered result:

1. **A twelve-iteration smoke run** of `fpo-policy default dppo cheetah` on
   HalfCheetah-v5, buffer 1,024, batch 64, on a laptop CPU: `rollout/reward`
   −277.1 to −188.9, `rollout/explained_variance` 0.0001 to 0.274,
   `losses/clipfrac` 0.0045 to 0.0070, checkpoints written, clean shutdown, no
   traceback. That is why a full run is worth attempting; it is not evidence
   about what a full run does.
2. **E6's FPO result**: −315 ± 26 to 1928 ± 224 over 500,000 steps, three
   seeds, and it did not fall back. Reference only. E6 ran on `cb5b369` and
   `main` has moved a long way since, so the FPO arm here is **E16's Phase A**,
   run on today's code, and not E6's published numbers.

---

## Held fixed

Identical to E16's Phase A, which is the FPO arm:

| | |
| --- | --- |
| Policy | `fpo-policy`, defaults, `--policy.device cpu` |
| Environment | `mujoco-v1`, `HalfCheetah-v5`, 1 env per client |
| Client | `--runner.replan-steps 1`, `--runner.seed $SEED` |
| Server seed | `--seed $SEED`, seeds 0, 1, 2 |
| Buffer | 4,096 → one update per 4,096 environment steps |
| Steps | 409,600 per seed = 100 updates |
| Concurrency | three runs at a time, alone on the machine |

Not fixed, and named so they are not mistaken for controlled: DPPO's
`cheetah` variant supplies `actor_lr` 1e-4, `critic_lr` 1e-3, both cosine
schedules with a ten-iteration warmup, `clip_ploss_coef` 0.01,
`update_epochs` 5, `batch_size` 2048, `gamma` 0.99, `sampling_noise_level`
0.1 and `use_normalized_rewards` true. Its own `buffer_size` of 20,000 is
overridden to 4,096 to match the FPO arm, which leaves two minibatches per
epoch. FPO's arm uses FPO's defaults.

Two fields in that variant do nothing and are not set here: `n_train_itr` and
`n_critic_warmup_itr`. The algorithm reads `train_itrs` and
`n_critic_warmup_itrs`, and the singular pair - carried over from DPPO's own
yaml naming - is dead while still appearing on the CLI. Named here because a
run configured through them would silently ignore them.

---

## What is measured

`rollout/reward` from each run's tensorboard, the scalar E6 reported. A run's
**value at a step** is the mean over the ten updates ending there, because
single updates in E6 swung by more than a thousand and no single update is a
result.

Three numbers per run: the value at its first ten updates (`start`), the
highest such value reached at any point (`best`), and the value at 409,600
(`end`).

---

## Predictions, and what falsifies each

**P1 — DPPO learns.** On at least **2 of 3** seeds, `end` exceeds `start` by
more than 500, which is about a fifth of the distance E6's FPO covered and far
outside the ±200 that single-update noise produces in the mean-of-ten.

> Falsified otherwise. Then PlugRL runs DPPO without DPPO learning here, which
> is a smaller claim and will be reported as exactly that.

**P2 — DPPO holds what it learns.** On at least **2 of 3** seeds, `end` is
within a factor of two of `best`, measured on the shift-to-positive scale
`value + 400` so that a ratio of returns that cross zero is meaningful.

> This is the property FPO fails on the bandit: four of ten runs in
> `e15-trust-region/results/toy/FINDINGS.md` end more than twice below their
> best. If DPPO holds where FPO does not, that is the first evidence that the
> failure is FPO's and not the setting's. If DPPO fails the same way, the
> failure belongs to something both share - the policy, the buffer, the
> platform - which is a more valuable answer than a win.

**P3 — the critic learns.** `rollout/explained_variance` ends above 0.5 on at
least 2 of 3 seeds.

> Falsified otherwise. A value head that cannot predict returns makes every
> advantage noise, and P1 and P2 would then be measuring a policy driven by
> noise rather than by DPPO.

**P4 — wall clock.** Under 150 minutes per seed, three concurrent. E16's Phase
A takes about 85 minutes for the same step count; DPPO's learn step is heavier
- five update epochs against FPO's defaults - so a bound rather than an
estimate.

> Falsified by exceeding it, and reported as a cost result whatever the
> returns do.

**No prediction is registered about which algorithm scores higher**, and none
will be read out of the comparison, for the reason in "What this cannot
settle" above.

---

## Declared deviations allowed in advance

1. **One restart** of any run that dies for a reason outside the experiment
   (a machine sleeping, a port collision, an unrelated process), recorded in
   `AMENDMENT.md` with the reason. A second such failure is reported, not
   worked around.
2. If a seed fails to reach 409,600 steps, it is dropped and the seed count
   for P1, P2 and P3 falls to what remains. Two seeds is the minimum for any
   reading; at one seed this reports only that it could not be run.

---

## Reading order

P4, then P3, then P1, then P2. P3 before P1 because a run whose critic never
learned cannot support a claim about what its policy did.
