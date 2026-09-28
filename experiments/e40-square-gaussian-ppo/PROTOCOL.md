# E40 measurement protocol (pre-registered)

**Written 2026-09-27, after the pilot in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

The coverage figure's Gaussian · PPO row (E38) has no square cell. From
random weights a Gaussian has nothing to learn from on square's sparse
reward (E27). DPPO fine-tunes its own Gaussian MLP with PPO on square, from
a pretrained checkpoint it releases, as the baseline to its diffusion
policy. #92 builds that policy and that setting.

**Run the way DPPO runs it, does its Gaussian MLP learn square under PPO
through PlugRL?**

### What this cannot settle

* Whether the CleanRL-style `gaussian-policy` of E38 would learn square. This
  is DPPO's Gaussian, a residual MLP of 1024 with chunks of 4, not that
  one.
* Comparison with DPPO's plotted numbers, which are deterministic
  evaluations under older settings (below).

---

## Declared in advance: what was already known

1. **DPPO's own Gaussian-MLP on square (state)**, from arXiv 2409.00588 Fig.
   7, read off the plot and approximate: about 0.3 at the start, about 0.8
   by about 5M steps, about 0.92 near 20M. These are deterministic
   evaluations. They were probably produced before v0.7 of the repository,
   whose config had an actor rate of 1e-5 and 1000 iterations. The config E40
   runs (cc7234ad) has 1e-4 and 201 iterations.
2. **E34**: DPPO's diffusion policy in DPPO's setting, the same client
   setting, 40 iterations. Training success went from 0.33-0.36 to
   0.65-0.70.
3. **The released checkpoint's deviation bound is 1.0, not the config's
   0.2.** The checkpoint's `logvar_max` replaces the config's value on load.
   DPPO's code does this, and #92 does the same.
4. **The pilot** (`results/pilot.txt`), seed 0:
   - Training success 0.250, then 0.235.
   - Explained variance −0.20 after the critic-only iteration, 0.44 after
     the next.
   - At the first actor update, approx_kl 0.0375 and clip fraction 0.78.
   - About 5.7 minutes an iteration.

---

## Design

One cell, `gauss-square`, on `guangzhao`, beside E39. Three seeds, each its
own server and two client processes of `robomimic-v1` on `square-img` (84 x
84 agentview, read by nothing). Episodes run 400 steps and do not end on
success. The client replans every 4 steps. **40 iterations** of 20,000
chunks (80,000 steps), 3.2M steps in all, as E34.

Server: `dppo-gaussian-policy default ppo dppo-square`, with
`--policy.checkpoint-path` pointing at the released checkpoint
(sha256 bf787a879dcaffe3) and nothing else changed:
- Policy: DPPO's Gaussian MLP, its `model` weights, σ from 0.1.
- Learning rates: actor 1e-4, critic 1e-3, constant.
- Minibatches of 10,000, 10 epochs.
- γ 0.999, λ 0.95, clip 0.01, target KL 1, value coefficient 0.5, no
  gradient clipping.
- Rewards scaled by a 0.99 running return and clipped at 10.
- One critic-only iteration.

Not DPPO's:
- A time-out ends the episode for GAE. DPPO's GAE bootstraps across the
  resets.
- There are no evaluation-only iterations.
- The data comes from 2 client processes, not 50 environments: the same
  20,000 chunks per iteration.
- The environments are not reset at every iteration.

Code: this branch (#92 plus this directory). Client: plugrl-env-client at
931ab56.

---

## Checks

* **V1 - the setting took**: every seed's server log shows the released
  checkpoint's `model` weights loaded with the deviation bounded at 1, and
  `PPOAlgoConfigDPPOSquare`. A seed that fails is not read.

---

## The status rule

E34's, with this cell's warmup. A seed's start is its mean
`rollout/success` over iterations 1-2, which the released policy collected
unchanged. Its end is the mean over iterations 31-40. The cell **learns** if
the end exceeds the start by at least **+0.2** on at least **2 of 3**
seeds.

---

## Predictions, and what falsifies each

**P1 - the cell runs end to end**: 40 iterations logged, at least 4
checkpoints, no traceback, clients exit 0, on every seed.

**P2 - DPPO's Gaussian MLP learns square under DPPO's own Gaussian PPO.**

> Grounds: known item 1. DPPO's own curve rises from about 0.3 to about 0.8
> by 5M steps, and E40 runs 3.2M at a rate ten times the one that curve
> probably used. Known item 2: E34's policy gained 0.29-0.37 on this budget.
> Against: the pilot's first update was clipped on 78% of samples.
> Falsified if fewer than two seeds gain 0.2.

**Reported, not predicted:**
- success, approx_kl and clip fraction by window of ten iterations
- wall clock

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, then the reported figures.
