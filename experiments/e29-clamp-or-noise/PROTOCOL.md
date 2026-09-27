# E29 measurement protocol (pre-registered)

**Written 2026-09-25, after the pilot in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

`fpo-policy` learns HalfCheetah, Hopper and Walker2d under FPO (E6, E16, E23,
E24) and none of them under DPPO (E17, E18, E28), where DPPO's own
`dppo-policy` improves all three (E28). By the project's rule that is a
defect of ours. Reading the code for where the two policies meet DPPO's loss
differently found two concrete differences:

1. **The clamp.** DPPO clamps every element's log-probability to [-5, 2]
   before forming the ratio. At the MuJoCo variants' noise level, 0.1, the
   upper bound binds on **50.7%** of `fpo-policy`'s elements - none at its
   first three steps, 93% at its last - and on none of `dppo-policy`'s. A
   clamped element carries no gradient and a ratio of exactly 1. #64 makes
   the bounds configurable.
2. **The noise.** `dppo-policy` floors every denoising step's deviation at
   the level; `fpo-policy`'s flow steps run from 0.1 down to about 0.01. Per
   element, the logged entropy is −1.920 for `fpo-policy` and 0.312 for
   `dppo-policy` - about a tenth of the deviation.

**Which of these stops `fpo-policy` learning under DPPO - either, both, or
neither?**

### What this cannot settle

* Any cause beyond these two. If neither arm learns, the remaining
  differences - network, chunk, number of denoising steps, output scale -
  are the next question, not this one's.
* Other tasks: E29 runs Hopper only, where E28's contrast is sharpest
  (`dppo-policy` +260 to +417, `fpo-policy` +5 to +11).

---

## Declared in advance: what was already known

1. **E28's `fpodppo-hopper`**, this experiment's cell unchanged: gain (mean of
   iterations 91-100 minus the first) +10.6, +4.9, +9.7; `approx_kl` around
   2.5 × 10⁻⁷ over the first ten iterations; entropy −1.920.
2. **The probe** (`clamp_probe.py`, `results/pilot.txt`): the clamp's reach at
   level 0.1 as above; at level 1.0, **no** `fpo-policy` element above 2
   and 0.19% below −5 (`dppo-policy` at 0.1: 0.17%), entropy 0.383.
3. **How far each actor moved in E28**, iteration 20 to 100, relative
   distance of the actor's weights, seeds 0 / 1 / 2: `fpo-policy` **22.5,
   20.0, 21.7%**, `dppo-policy` 11.9, 12.0, 11.6%. `fpo-policy`'s weights do move - further
   than `dppo-policy`'s - while its action distribution barely changes.
4. **The pilot** (`results/pilot.txt`), eleven iterations, seed 0: every arm
   ran; each server logged the configuration intended; entropies −1.9198,
   −1.9198, +0.3828. Against `control`, `noclamp` raised the mean
   `approx_kl` 2.4-fold (5.9 × 10⁻⁷) and the actor's gradient norm 1.5-fold;
   `wide` lowered `approx_kl` tenfold (2.7 × 10⁻⁸) - at an entropy close to
   `dppo-policy`'s, whose `approx_kl` is about 10⁻⁵. No arm's return moved
   in eleven iterations.

---

## Design

| arm | extra algorithm flags | the clamp | the noise |
| --- | --- | --- | --- |
| `control` | none | binds on 50.7% | level 0.1 |
| `noclamp` | `--algo.logprob-clamp-max inf` | lifted | level 0.1 |
| `wide` | `--algo.sampling-noise-level 1.0 --algo.logprob-noise-level 1.0` | cannot bind | level 1.0, entropy ≈ `dppo-policy`'s |

Every arm is E28's `fpodppo-hopper` cell: `fpo-policy`, `dppo hopper`,
Hopper-v5, buffer 4,096, the variant's minibatch of 2,048, 100 iterations,
seeds 0, 1, 2, checkpoints every 20 iterations. `run_cell.sh` is E24's with
one addition, `EXTRA_ALGO_ARGS`. `wide` raises the level for both sampling
and the log-probability, so the ratio is taken under the distribution the
actions were drawn from. Only the upper bound is lifted in `noclamp`: it is
the one that binds.

`wide` changes the noise and, as a consequence, removes the clamp's reach
(known item 2); `noclamp` removes the clamp's reach alone. The design cannot
add noise while keeping the clamp's reach, so "the noise alone" is read
from the two together, below.

Machine `guangzhao`, CPU, one thread per process, the three arms at once.
Code: `main` plus #64, run from this branch's worktree through `PYTHONPATH`;
client plugrl-env-client at 931ab56.

---

## Checks

* **V1 - the manipulation took.** Each arm's logged `losses/entropy`, which
  depends only on the noise schedule, is within 0.005 of **−1.920**
  (`control`, `noclamp`) and **−1.920 + ln 10 = +0.383** (`wide`) at every
  iteration. An arm that fails V1 is not read.

---

## Predictions, and what falsifies each

The gain is the mean return of iterations 91-100 minus the first
iteration's.

**P1 - every arm runs end to end**: 100 iterations logged, the checkpoint at
100, no traceback, client exit 0, on every seed.

**P2 - the control reproduces E28**: `control`'s gain is at most **+50** on
every seed.

> E28: +5 to +11. Falsified: the control moved, and nothing below is read.

**P3 - lifting the clamp alone lets `fpo-policy` learn**: `noclamp`'s gain is
at least **+100** on at least 2 of 3 seeds.

**P4 - ten times the noise lets it learn**: `wide`'s gain is at least **+100**
on at least 2 of 3 seeds.

> +100 is about nine times the control's largest gain in E28 and under half
> of `dppo-policy`'s smallest (+260).

**What I expect, after the pilot: both P3 and P4 falsified.** In eleven
iterations `noclamp` changed the update little and `wide` made the ratio
move less, not more. The hypotheses are registered as the claims they would
support, so that a falsification rules each one out.

**Reading P3 and P4 together:**

| P3 `noclamp` | P4 `wide` | reading |
| --- | --- | --- |
| holds | holds | the clamp is enough to stop it; the noise need not change |
| holds | falsified | the clamp stops it, and ten times the noise hurts in some other way |
| falsified | holds | the clamp alone is not it; the narrow noise is - with or without the clamp |
| falsified | falsified | neither: both are ruled out as sufficient causes |

**Reported, not predicted:** `approx_kl` and `clipfrac` over the first and
last ten iterations; the actor's relative movement from iteration 20 to 100,
beside E28's; each run's return and episode length.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.
2. A failure caused by how an arm was launched is fixed and the arm rerun
   once, recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, P2, then P3 and P4 and the table, then the reported figures.
