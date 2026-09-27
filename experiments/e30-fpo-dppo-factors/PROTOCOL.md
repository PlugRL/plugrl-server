# E30 measurement protocol (pre-registered)

**Written 2026-09-27, after the pilot in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

`fpo-policy` learns Hopper under FPO and barely moves under DPPO (E28), where
`dppo-policy` gains +260 to +417 in the same 100 iterations. E29 ruled out
DPPO's log-probability clamp and found that ten times the noise helps but
reaches only about a quarter of `dppo-policy`'s gain. Three differences
between the two policies remain untested:

| | `fpo-policy` | `dppo-policy` (hopper) |
| --- | --- | --- |
| actions per inference | 1 | 4 |
| denoising steps | 10 | 20 |
| network | 4 hidden layers of 32 | 3 hidden layers of 512, residual |

**Which of them - if any - closes the gap?** One arm per difference, on top
of E29's `wide` (noise level 1.0), and one with all three.

### What this cannot settle

* Why a difference matters, if one does. Each arm changes one knob.
* Whether the answer holds at the default noise level: every arm runs at
  level 1.0, E29's best.
* Other tasks. Hopper only, where the contrast is sharpest.

---

## Declared in advance: what was already known

1. **E29's `wide`**, this experiment's `base` arm unchanged: gain (mean of
   iterations 91-100 minus the first) +73.6, +84.6, +106.2; entropy +0.383;
   `approx_kl` about 2.5 x 10⁻⁸ over its first ten iterations.
2. **E28's `dppo-policy` on Hopper**: gain +260 to +417; `approx_kl` about
   10⁻⁵ and `clipfrac` 1-2%.
3. **The pilot** (`results/pilot.txt`), eleven iterations, seed 0: every arm
   ran. `net512` alone had updates of `dppo-policy`'s size (`approx_kl`
   1.0 x 10⁻⁵, `clipfrac` 1.4%) and a return that rose, 18.1 to 60.6; the
   other four stayed at `approx_kl` 10⁻⁸ to 10⁻⁶ and returns of 17 to 26.
   Twenty flow steps lower the entropy to -0.004.

---

## Design

| arm | on top of `wide` | buffer / batch | client replans every |
| --- | --- | --- | --- |
| `base` | nothing | 4,096 / 2,048 | 1 |
| `chunk4` | `--policy.action-horizon 4` | 1,024 / 512 | 4 |
| `steps20` | `--policy.flow-steps 20` | 4,096 / 2,048 | 1 |
| `net512` | `--policy.hidden-dims 512 512 512` | 4,096 / 2,048 | 1 |
| `all` | all three | 1,024 / 512 | 4 |

Every arm: `fpo-policy`, `dppo hopper`, Hopper-v5, noise level 1.0 for
sampling and the log-probability, 100 iterations of 4,096 environment steps,
seeds 0, 1, 2, checkpoints every 20 iterations. With a chunk of 4 the buffer
holds 1,024 entries of 4 steps each, so an iteration is still 4,096 steps
and an epoch still two minibatches, as for `dppo-policy`. `base`, `chunk4` and
`steps20` run first; `net512` and `all` after. Machine `guangzhao`, CPU, one
thread per process; code `main` (0d29c8c), client 931ab56.

---

## Checks

* **V1 - the manipulation took.** Logged `losses/entropy`, at every
  iteration, within 0.005 of **+0.383** (`base`, `chunk4`, `net512`) and of
  **-0.004** (`steps20`, `all`, the pilot's value). An arm that fails is not
  read.

---

## Predictions, and what falsifies each

The gain is the mean return of iterations 91-100 minus the first
iteration's. An arm **closes most of the gap** if its gain is at least
**+200** on at least 2 of 3 seeds: above E29's best seed by nearly 100, and
within reach of `dppo-policy`'s smallest (+260).

**P1 - every arm runs end to end** (100 iterations, the checkpoint at 100,
no traceback, client exit 0, every seed).

**P2 - the network closes most of the gap**: `net512` does.

> Grounds: the pilot. Falsified if fewer than two of its seeds reach +200.

**P3 - neither the chunk nor the step count does**: `chunk4` and `steps20`
do not.

> Grounds: the pilot, and E28's `fpo-policy` never moving. Falsified if
> either closes most of the gap.

**Reported, not predicted:** `base` (E29 again) and `all`; for every arm
its gain, `approx_kl` and `clipfrac`, and whether it meets the coverage
figure's Hopper bar - the mean of iterations 91-100 at least **500** on 2 of
3 seeds, E28's status rule.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.
2. A failure caused by how an arm was launched is fixed and the arm rerun
   once, recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, P2, P3, then the reported figures.
