# E38 measurement protocol (pre-registered)

**Written 2026-09-27, before any E38 run. The three-iteration pilot and the
GAE diagnostic below came first and are recorded in `results/pilot.txt`.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

The coverage figure's rows are expressive policies with the algorithms built
for them. It has no row for the baseline they are measured against. #85 adds
one: `gaussian-policy` under `ppo`, written to CleanRL's
`ppo_continuous_action.py` with every default.

**Run as CleanRL runs it, does it learn HalfCheetah, Hopper and Walker2d
through PlugRL?**

### What this cannot settle

* Whether PlugRL reproduces CleanRL's returns. The tasks are -v5, not -v4,
  and three things happen differently (declared below). The returns are
  reported beside CleanRL's, not tested against them.
* Anything about the other rows.

---

## Declared in advance: what was already known

1. **CleanRL's own results** for this file, from its documentation (MuJoCo
   -v4; the benchmark command passes no `--total-timesteps`, so its default
   of 1,000,000; three seeds): HalfCheetah 1442.64 +/- 46.03, Hopper
   2382.86 +/- 271.74, Walker2d 2287.95 +/- 571.78.
2. **The pair on a bandit** (#85's tests, with #86 merged): from about
   -0.78 to between -0.120 and -0.160 on five seeds, final over best
   1.00-1.14.
3. **The pilot**: three iterations on each task, seed 0, at ab6e32f. Every
   run ended with exit 0 and no traceback, at 528-584 environment steps a
   second including learning - about 30 minutes for 488 iterations alone.
4. **GAE ended every episode one step late** on every branch since 48042e5
   (#86); E38 runs with the fix. A diagnostic of this pair on Hopper-v5, 100
   iterations (annealed over 100), seeds 0-2, at ab6e32f and 274929c:
   iterations 41-50 at 267 / 589 / 367 before the fix and 674 / 673 / 454
   after; iterations 91-100 at 432 / 1512 / 910 before and 1005 / 868 / 949
   after.
5. **Runs are not reproducible bit for bit** after the first update (E20).

### Where this differs from CleanRL, declared

* Observation statistics are updated after each iteration's learning, from
  that iteration's observations, not at every step; the first iteration runs
  unnormalised. CleanRL's NormalizeObservation updates at every step from
  the first.
* Rewards are scaled by the return's running deviation once per iteration,
  over the iteration's returns, not step by step as NormalizeReward does.
* The environments are -v5.

---

## Design

| cell | task | seeds | iterations |
| --- | --- | --- | --- |
| `ppo-cheetah` | HalfCheetah-v5 | 0, 1, 2 | 488 |
| `ppo-hopper` | Hopper-v5 | 0, 1, 2 | 488 |
| `ppo-walker` | Walker2d-v5 | 0, 1, 2 | 488 |

`gaussian-policy default ppo default` with the task's dimensions and nothing
else: rollouts of 2,048 steps, minibatches of 64, ten epochs, clip 0.2,
learning rate 3e-4 annealed to zero over the 488 iterations (999,424 steps,
CleanRL's 1,000,000 // 2,048). One environment per seed, the client
replanning every step; checkpoints every 20 iterations. E30's
`run_cell.sh`, unchanged, through `run.sh`.

Machine `guangzhao`, CPU, one thread per process, the three cells at once
beside E37; code this branch (#85 and #86 merged, plus this directory),
client plugrl-env-client at 931ab56.

---

## Checks

* **V1 - the configuration took**: every iteration's logged
  `models/learning_rate` equal to (1 - (i-1)/488) x 3e-4 for iteration i,
  to one part in a million. A cell that fails is not read.

---

## The status rule

As the coverage figure has it, on the mean return of iterations 479-488, on
at least **2 of 3** seeds: Hopper and Walker2d at least **500**; HalfCheetah
at least **+200** over its first iteration. A cell that passes **learns**;
one that does not **did not learn in 999,424 steps**.

---

## Predictions, and what falsifies each

**P1 - every cell runs end to end** (488 iterations logged, at least 24
checkpoints, no traceback, client exit 0, on every seed).

**P2 - it learns Hopper.** **P3 - it learns Walker2d.** **P4 - it learns
HalfCheetah.**

> Grounds for all three: known item 1 - CleanRL's returns with this
> configuration are four to five times the bars on Hopper and Walker2d, and
> 1443 on HalfCheetah, where an untrained policy starts near -300 (E6).
> For Hopper also known item 4: with the fix, 868 to 1005 by iteration 100.
> Each is falsified if fewer than two seeds clear its bar.

**Reported, not predicted:** the mean return over iterations 91-100, 241-250
and 479-488 for every seed, beside CleanRL's; wall clock.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, P3, P4, then the reported figures.
