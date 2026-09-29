# E37 measurement protocol (pre-registered)

**Written 2026-09-27, after the pilots in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

`fpo-policy`'s two robomimic square cells, under FPO and under DPPO, have
only ever started from random weights (E27): success 0 throughout, as it is
for any policy that starts there on square's sparse reward. Neither
algorithm's authors fine-tune square that way. DPPO fine-tunes a policy
pretrained on demonstrations; FPO++ fine-tunes behaviour-cloned flow policies.
E35 cloned `fpo-policy` on the 300 demonstrations DPPO pretrained on.

**Started from that clone, does `fpo-policy` learn square under FPO, and
under DPPO?**

### What this cannot settle

* Comparison with either paper's numbers. The success read here is that of
  the training rollouts, with each algorithm's sampling; neither setting is
  reproduced in full (Design).
* Whether a better clone would change the answer.

---

## Declared in advance: what was already known

1. **The clone** (E35): `fpo-policy`, chunks of 4, three hidden layers of
   1024, 400,000 steps of flow matching on the demonstrations; fifty
   evaluation episodes each: success 0.44 at 10 flow steps, 0.38 at 20.
2. **E34** runs `dppo-policy` from DPPO's own checkpoint in the same
   environment; its start was 0.29 to 0.35 in its pilot.
3. **The first pilot** (`results/pilot.txt`): with the observation statistics
   updating, FPO's critic-only iteration showed a mean ratio of 1.38 and the
   next iteration's success fell from 0.51 to 0.39; #82 freezes them, and
   both cells run with it.
4. **The second pilot**, statistics frozen, one seed, one client: FPO's
   critic-only iteration gave a ratio of exactly 1.000 and success 0.49 then
   0.50; its first update, ratio 0.993 with 43% clipped. DPPO's three
   iterations by the unchanged clone succeeded 0.455, 0.295, 0.365 - swings
   larger than binomial noise, as in E34 (0.27 to 0.40 between unchanged
   iterations) - and its first update had `approx_kl` 0.00023 and `clipfrac`
   0.10. An iteration takes about 4 minutes for FPO and 13 for DPPO with three
   clients.

---

## Design

Two cells on `guangzhao`, three seeds each, run at once, each seed its own
server and three client processes of `robomimic-v1` on `square-img` (84 x 84
agentview, read by nothing), episodes of 400 steps that do not end on
success, replanning every 4 steps. Both start from E35's clone
(`--algo.restore except-critic`: the value head from the run's own
initialisation) with its observation statistics frozen.

| cell | algorithm | iterations | steps per iteration |
| --- | --- | --- | --- |
| `fpo-square` | FPO with FPO++'s square fine-tuning, as far as this FPO has it | 60 | 48,000 |
| `dppo-square` | `dppo square` (#77) with E33's changes for a flow policy | 40 | 80,000 |

`fpo-square`: FPO++'s chunk loss over the 4 executed steps and 7 dimensions,
velocity error, uniform flow times, one ratio per sample, one critic-only
iteration, clip 0.01, learning rate 1e-5, 10 epochs of 8 minibatches, 8
samples per action, gamma 0.995, lambda 0.99, 12,000 chunks per iteration, 10
flow steps. Not FPO++'s: one learning rate for actor and critic (FPO++: 1e-5
and 1e-4), squared error rather than Huber, no gradient clipping, whole-buffer
advantage normalisation, the clip of the log-ratio at 5 but none on the old
loss, and each seed's 48,000 steps collected by 3 client processes (FPO++: 30
environments).

`dppo-square`: every value of DPPO's square config (#77), with E33's noise
level 1.0 for sampling and the log-probability and 20 flow steps - the
setting under which `fpo-policy` learns all three MuJoCo tasks under DPPO -
and minibatches of 500 entries, which are DPPO's 10,000 (chunk, step)
samples at 20 steps each. Not DPPO's: all 20 steps are fine-tuned (DPPO
fine-tunes 10 of a diffusion policy's 20; `fpo-policy` has no frozen copy).

Code: this branch - #74, #77, #82 and E35's scripts merged in; client
plugrl-env-client at 931ab56.

---

## Checks

* **V1 - the settings took**: every seed's server log restores the clone
  with `except-critic` and shows its cell's settings (`summarise.py` lists
  them) and `freeze_obs_stats=True`.

---

## The status rule

A seed's **start** is its mean `rollout/success` over the iterations its
cloned policy collected unchanged: 1-2 for `fpo-square` (one critic-only
iteration), 1-3 for `dppo-square` (two). Its **end** is the mean over its
last ten iterations. A cell **learns** if the end exceeds the start by at
least **+0.2** on at least **2 of 3** seeds - E34's rule.

---

## Predictions, and what falsifies each

**P1 - both cells run end to end**: every iteration logged, the
checkpoints, no traceback, client exit 0, on every seed.

**P2 - `fpo-policy` learns square under FPO++'s fine-tuning.**

**P3 - `fpo-policy` learns square under DPPO's.**

> Grounds for both: each paper fine-tunes square from a cloned or
> pretrained policy of about this success and gains more than 0.2; E33 shows
> this policy class learns under DPPO in this setting. Falsified if fewer
> than two seeds gain 0.2.

**Reported, not predicted:** each seed's success per window of ten
iterations.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, P3, then the reported figures.
