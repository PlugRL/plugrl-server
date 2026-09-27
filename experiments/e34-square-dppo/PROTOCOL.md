# E34 measurement protocol (pre-registered)

**Written 2026-09-27, after the pilot in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E27 ran every policy-algorithm pair on robomimic's square task from random
weights with its sparse reward, and could only say each runs: success was 0
throughout. That is not how DPPO is used on square. DPPO fine-tunes a
released pretrained diffusion policy, through the last 10 of its 20
denoising steps, with its own settings - and reports success rising from
about 0.4 to about 0.8 in 2.5 to 3 million environment steps (its Fig. 5 and
7, deterministic evaluations).

#73, #76 and #77 give `dppo-policy` and `dppo` that setting: the released
checkpoint's `ema` weights loaded strictly, the last 10 of 20 steps
fine-tuned through a frozen copy of the first 10, and every value of DPPO's
`cfg/robomimic/finetune/square/ft_ppo_diffusion_mlp.yaml`.

**In DPPO's own setting, does `dppo-policy` learn square on PlugRL?**

### What this cannot settle

* Comparison with DPPO's numbers. DPPO reports deterministic evaluations;
  this reads the success of the training rollouts, which carry DPPO's
  sampling noise (a deviation of at least 0.1 per step).
* Anything about `fpo-policy` on square, which has no pretrained policy.
* The full schedule: DPPO runs 201 iterations (16 million steps); this runs
  40.

---

## Declared in advance: what was already known

1. **The pilot** (`results/pilot.txt`): one seed, three iterations, all
   collected by the released policy unchanged - success 0.313, 0.288,
   0.345; the one policy update had `approx_kl` 0.0016 and `clipfrac` 0.19;
   about 7 minutes per iteration.
2. **The released checkpoint** loads strictly and equals its `ema` weights
   (#73's check); PlugRL's square normalisation file is byte-identical to
   DPPO's.
3. **E27**: square from random weights, 0 success in every iteration.

---

## Design

`dppo-policy square dppo square`, `--policy.checkpoint-path` the released
`state_8000.pt` (sha256 368ce587...), 40 iterations, seeds 0, 1, 2, run at
once; each seed its own server and six client processes of `robomimic-v1` on
`square-img` (84 x 84 agentview, read by nothing), episodes of 400 steps that
do not end on success (`--env.no-terminate-on-success`), replanning every 4
steps; checkpoints every 10 iterations. An iteration is 20,000 chunks - 80,000
environment steps, 200 episodes per seed. Machine `guangzhao`, CPU, one
thread per process. Code: this branch (#77 plus this directory); client
plugrl-env-client at 931ab56.

Known differences from DPPO's run, declared: PlugRL shuffles buffer entries
(each with its 10 steps) where DPPO shuffles single (chunk, step) samples;
PlugRL clips unnormalised actions to the data's range; the client is not
seeded (robomimic-v1 refuses a seed).

---

## Checks

* **V1 - the setting took**: every seed's server log shows
  `ft_denoising_steps=10` and "Loaded the ema weights of" the checkpoint. A
  seed that fails is not read.

---

## The status rule

The **start** is a seed's mean `rollout/success` over iterations 1-3, all
collected by the released policy unchanged (the first two iterations train
only the critic). The **end** is its mean over iterations 31-40. The cell
**learns** if the end exceeds the start by at least **+0.2** on at least **2
of 3** seeds; otherwise it **did not learn in 40 iterations**.

---

## Predictions, and what falsifies each

**P1 - the cell runs end to end**: 40 iterations logged, at least 4
checkpoints, no traceback, client exit 0, on every seed.

**P2 - `dppo-policy` learns square under DPPO's own fine-tuning.**

> Grounds: DPPO's curves rise by about 0.4 by 2.5 to 3 million steps, and
> 40 iterations are 3.2 million; the pilot's first update moved the policy
> at DPPO's usual scale. Falsified if fewer than two seeds gain 0.2.

**Reported, not predicted:** each seed's success, return and episode length
per window of ten iterations; `approx_kl` and `clipfrac`.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, then the reported figures.
