# E40: DPPO's Gaussian MLP learns robomimic square under DPPO's own Gaussian PPO, through PlugRL

2026-09-27/28 · Linux workstation (`guangzhao`), CPU only · one cell, three
seeds, 40 iterations of 80,000 steps · protocol: [`PROTOCOL.md`](PROTOCOL.md)
(`456da16`, after the pilot in `results/pilot.txt`, before the registered
run)

---

## The result

`dppo-gaussian-policy`, started from DPPO's released square Gaussian
(`square_pre_gaussian_mlp_ta4`, its `model` weights), under `ppo
dppo-square`, which is every value of DPPO's `ft_ppo_gaussian_mlp.yaml` (#92).
Training-rollout success:

| seed | start: iterations 1-2 | end: iterations 31-40 | gain |
| --- | --- | --- | --- |
| 0 | 0.185 | 0.547 | **+0.362** |
| 1 | 0.195 | 0.537 | **+0.342** |
| 2 | 0.212 | 0.565 | **+0.352** |

* **P1 holds**: 40 iterations on every seed, four checkpoints each, no
  traceback, clients exited 0; 4 hours, beside E39.
* **V1 passes**: every server loaded the released checkpoint's `model`
  weights, logged the deviation bounded at 1, and ran
  `PPOAlgoConfigDPPOSquare`.
* **P2 holds**: it learns square, 3 of 3 seeds past +0.2.

The Gaussian · PPO row's square cell, empty until now, **learns**. That makes
Gaussian · PPO the third pair to learn all four tasks. The other two are both
DPPO pairs: `dppo-policy` · DPPO and, since E37, `fpo-policy` · DPPO.

---

## The curves

Success by window of ten iterations:

| seed | 1-10 | 11-20 | 21-30 | 31-40 |
| --- | --- | --- | --- | --- |
| 0 | 0.273 | 0.422 | 0.454 | 0.547 |
| 1 | 0.256 | 0.394 | 0.504 | 0.537 |
| 2 | 0.310 | 0.402 | 0.506 | 0.565 |

Still rising at 40. Mean return, one per step spent succeeding, went from
30-36 over the first two iterations to 109-114 over the last ten.
`approx_kl` averaged 2 to 6 x 10⁻³ and the clip fraction about 0.45
throughout. A clip of 0.01 on the ratio of the chunk's mean log-probability
binds on almost half the samples at every update, as the pilot's first
update suggested. DPPO's own setting runs that way, and it learns anyway.

---

## What E40 does not show

* **That it reproduces DPPO's curve.** These are training rollouts with the
  policy's deviation, from a start of about 0.2, where DPPO plots
  deterministic evaluations from about 0.3. The rates are the current
  config's (1e-4), not the ones the paper's plot probably used, and GAE
  here ends an episode at its time-out where DPPO's bootstraps across it.
* **Anything about E38's `gaussian-policy`.** This is DPPO's Gaussian, a
  residual MLP of 1024 with chunks of 4, not CleanRL's 64 x 64.

Checkpoints and tensorboards stay on `guangzhao`. The logs, `verdicts.txt`
and `summary.tsv` are here.
