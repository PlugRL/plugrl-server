# E28: under DPPO, `dppo-policy` learns HalfCheetah and climbs Hopper and Walker2d; `fpo-policy` does not move

2026-09-25 · Linux workstation (`guangzhao`), CPU only · five DPPO cells,
three seeds each, 100 iterations of 4,096 steps · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`d1f4783`, before any run)

---

## The result

E24 ran every DPPO combination on MuJoCo for twenty iterations and claimed
only that each runs. E28 ran the same five cells, unchanged, five times as
long, with a rule for what "learns" means fixed in advance.

| cell | mean of iterations 91-100, seeds 0 / 1 / 2 | status |
| --- | --- | --- |
| `dppo-policy` · DPPO · HalfCheetah | **+481 / +84 / +544** over the first iteration | **learns** (2 of 3 past +200) |
| `dppo-policy` · DPPO · Hopper | 339 / 430 / 276, from 11-16 | did not learn in 409,600 steps (0 of 3 past 500) |
| `dppo-policy` · DPPO · Walker2d | 260 / 292 / 274, from −3 to −2 | did not learn in 409,600 steps (0 of 3) |
| `fpo-policy` · DPPO · Hopper | 25 / 25 / 27, from 15-20 | did not learn in 409,600 steps (0 of 3) |
| `fpo-policy` · DPPO · Walker2d | 3 / 1 / 2, from −0 to 2 | did not learn in 409,600 steps (0 of 3) |

* **P1 holds, 5 of 5**: all fifteen seeds logged 100 iterations, wrote the
  checkpoint at 100, have no traceback and a client that exited 0; 15 to 20
  minutes per cell.
* **P2 falsified**: `dppo-policy` does not reach 500 on Walker2d on any seed.
* **P3 falsified**: nor on Hopper.
* **P4 holds**: `fpo-policy` under DPPO clears neither threshold, by far.
* **Reported**: `dppo-policy` learns HalfCheetah by the rule, on seeds 0 and
  2.

With E24 and E27, the matrix of the two MLP policies now reads:

| | HalfCheetah | Hopper | Walker2d | robomimic square |
| --- | --- | --- | --- | --- |
| `fpo-policy` · FPO | learns (E6, E16) | learns (E23) | learns (E24) | runs (E27) |
| `fpo-policy` · DPPO | runs, no learning in 409,600 steps (E17, E18) | **runs, no learning (E28)** | **runs, no learning (E28)** | runs (E27) |
| `dppo-policy` · DPPO | **learns (E28)** | **runs, rising: 276-430 (E28)** | **runs, rising: 260-292 (E28)** | runs (E27) |

---

## Where my predictions went wrong

P2 and P3 extrapolated E24's twenty-iteration slope, and neither curve kept
it. Mean return over iterations 11-20, 41-50 and 91-100, seeds 0 / 1 / 2:

| cell | 11-20 | 41-50 | 91-100 |
| --- | --- | --- | --- |
| `dppo-policy` · Walker2d | 186 / 172 / 155 | 251 / 224 / 267 | 260 / 292 / 274 |
| `dppo-policy` · Hopper | 77 / 74 / 80 | 195 / 214 / 201 | 339 / 430 / 276 |
| `dppo-policy` · HalfCheetah | −382 / −394 / −403 | −332 / −379 / −331 | 95 / −361 / 162 |

Walker2d flattened after about fifty iterations. Hopper was still climbing
at the end - seed 1's best iteration reached 493. HalfCheetah, which showed
nothing in twenty iterations and was registered without a prediction, is the
one that crossed its threshold.

Every seed's first iteration is identical to E24's, to the decimal: the
collection before the first update is deterministic, as E20 found, and
what follows is not (known item 4).

---

## What the contrast shows, and what it does not

The protocol's joint reading - "the policy class decides whether DPPO learns
from a random start" - required P2 or P3 to hold, and neither did, so it is
not claimed. Read as description, with every setting held equal - the same
algorithm and variant, 4,096 environment steps per iteration, two minibatches
per epoch and so five optimiser steps per iteration (the accumulation window
of 8 never fills, and each epoch's end forces the step) - the two policy
classes do not behave alike at all:

| | HalfCheetah | Hopper | Walker2d |
| --- | --- | --- | --- |
| `dppo-policy`, last ten minus first | +84 to +544 | +260 to +417 | +262 to +294 |
| `fpo-policy`, the same | +2 to +20 (E18; last ten minus iterations 1-10) | +5 to +11 | +1 to +2 |

`fpo-policy` learns all three tasks under FPO (E6, E16, E23, E24) and none
under DPPO. By the project's rule, that is a defect of ours to locate, not a
finding about DPPO: the same policy is trainable, and the same algorithm
trains a different policy.

**One candidate, from reading the code, not tested.** The two policies set
each denoising step's noise differently. `dppo-policy` clamps each step's
standard deviation from below at the variant's `sampling_noise_level`, 0.1
here, so every step is at least 0.1. `fpo-policy`, a flow policy, uses
σ_t = level · √(t / (1 − t)) and a step deviation of σ_t · √dt, which over its
ten steps is 0.1 at the first, about 0.03 midway and about 0.01 at the last.
E18 found `approx_kl` near 10⁻⁸ and `clipfrac` at 0 for `fpo-policy` under
DPPO - the ratio barely leaves 1. Whether the noise schedule is why is the
next controlled comparison, not something E28 can say.

---

## What this does and does not support

**Supported:**

* `dppo-policy`, trained by DPPO from a random initialisation on this
  platform, learns HalfCheetah-v5 within 409,600 steps on two of three seeds,
  and improves Hopper-v5 and Walker2d-v5 on every seed.
* `fpo-policy` under DPPO does not learn Hopper or Walker2d in 409,600 steps,
  as it did not learn HalfCheetah (E17, E18).

**Not supported:**

* That DPPO learns Hopper or Walker2d on this platform: neither reached the
  bar set in advance.
* Any reason for the contrast. The two policy classes differ in network,
  action chunk, denoising steps and noise; E28 holds the algorithm fixed and
  varies all of those at once.
* Anything about DPPO as published, which fine-tunes pretrained policies.

---

## Reproducing

```bash
bash run.sh         # five cells in two waves; about 37 minutes on 24 cores
python summarise.py # P1, the status rule, P2-P4, the figures; writes summary.tsv
```

`run.sh` calls E24's `run_cell.sh` unchanged. `results/verdicts.txt` is
`summarise.py`'s output; `summary.tsv` has one row per cell and seed in E24's
columns, with `claim` set from each cell's status. Checkpoints and tensorboard
files stay on `guangzhao` (`.gitignore`).
