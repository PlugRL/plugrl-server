# E24: every MuJoCo combination the code allows runs, and FPO learns Walker2d

2026-09-25 · Linux workstation (`guangzhao`), CPU only · two policies, two
algorithms, three MuJoCo tasks, three seeds per cell · protocol:
[`PROTOCOL.md`](PROTOCOL.md) · amendment: [`AMENDMENT.md`](AMENDMENT.md) -
moved off the Windows machine before any cell finished, and rerun from the start

---

## The result

PlugRL keeps the policy, the algorithm that trains it and the environment
apart. Before E24, three combinations of the two MLP policies and two
algorithms had run on MuJoCo, and DPPO's own policy class, `dppo-policy`, had
never run on this platform at all. E24 filled the rest of that block.

**All six cells run end to end, and FPO learns Walker2d on every seed.**

| cell | policy · algorithm · task | claim | result |
| --- | --- | --- | --- |
| `fpo-walker` | `fpo-policy` · FPO · Walker2d-v5 | learns | **964, 1027, 1101** from −0.3 to 0.7 |
| `fpodppo-hopper` | `fpo-policy` · DPPO · Hopper-v5 | runs | runs, 3 of 3 |
| `fpodppo-walker` | `fpo-policy` · DPPO · Walker2d-v5 | runs | runs, 3 of 3 |
| `dppo-cheetah` | `dppo-policy` · DPPO · HalfCheetah-v5 | runs | runs, 3 of 3 |
| `dppo-hopper` | `dppo-policy` · DPPO · Hopper-v5 | runs | runs, 3 of 3 |
| `dppo-walker` | `dppo-policy` · DPPO · Walker2d-v5 | runs | runs, 3 of 3 |

**P1 holds, 6 of 6**: on all eighteen seeds every iteration is logged, the
checkpoint is written, no server log holds a traceback and every client
exits 0. **P2 holds, 3 of 3**: the mean of iterations 91-100 of FPO on
Walker2d is 964.0, 1026.5 and 1101.0, against the threshold of 500 and an
untrained first iteration of −0.3 to 0.7; episodes grew from about 19 steps to
about 415.

With E6, E16 and E23 before it, the MuJoCo block now reads:

| | HalfCheetah | Hopper | Walker2d |
| --- | --- | --- | --- |
| `fpo-policy` · FPO | learns (E6, E16) | learns (E23) | **learns (E24)** |
| `fpo-policy` · DPPO | runs (E17, E18) | **runs (E24)** | **runs (E24)** |
| `dppo-policy` · DPPO | **runs (E24)** | **runs (E24)** | **runs (E24)** |
| `dppo-policy` · FPO | does not exist: FPO needs a flow policy | | |

---

## One thing the run-only cells showed, unregistered

No DPPO cell claimed to learn: E17 and E18 found DPPO did not, from a random
initialisation, in `fpo-policy`. Read as description, the twenty iterations
here say that finding belongs to `fpo-policy`, not to DPPO:

| cell | return, first → mean of last ten | episode length |
| --- | --- | --- |
| `fpo-policy` · DPPO · Hopper | 14.9, 19.7, 17.0 → 15.7, 17.6, 18.1 | ~21 → ~22 |
| `fpo-policy` · DPPO · Walker2d | 1.6, 0.0, 0.5 → 1.4, 0.1, 1.1 | ~20 → ~20 |
| `dppo-policy` · DPPO · Hopper | 11.2, 12.4, 15.8 → **75.7, 77.2, 81.5** | ~19 → **~50** |
| `dppo-policy` · DPPO · Walker2d | −2.2, −1.6, −2.6 → **220.0, 191.9, 156.5** | ~17 → **~194** |
| `dppo-policy` · DPPO · HalfCheetah | −386.3, −445.0, −382.1 → −384.0, −365.7, −418.2 | 1,000 throughout |

Under the same algorithm, the same buffer of 4,096 environment steps and the
same twenty iterations, DPPO's own policy class starts to learn Hopper and
Walker2d on every seed, and `fpo-policy` does not move. That fits E18's
untested candidate: `fpo-policy` denoises all ten steps under one small
sampling noise, while DPPO's policy clamps each step's noise from below and
predicts a chunk of four actions. It does not test it. Twenty iterations is
also far too short to call anything learned, and HalfCheetah shows nothing
either way.

---

## What this does and does not support

**Supported:**

* On this platform, both MLP policies run under DPPO on HalfCheetah, Hopper
  and Walker2d, and `fpo-policy` runs under FPO on all three - every
  combination of these policies, algorithms and tasks the code allows.
* FPO learns Walker2d-v5 from a random initialisation, to 964-1,101 in
  409,600 steps on three seeds, with defaults tuned for nothing in
  particular.
* `dppo-policy`, DPPO's own policy class with its bundled D4RL configurations
  and normalisation, runs end to end here for the first time.

**Not supported:**

* That DPPO learns any of these tasks. No DPPO cell was registered to learn,
  and twenty iterations cannot show it. The early rise in two
  `dppo-policy` cells is a reason to run them longer, not a result.
* Anything about wall-clock cost: the machine changed mid-experiment, and
  every process was capped at one thread (amendment 1).
* Anything outside the MuJoCo block. robomimic, which `dppo-policy` also has a
  configuration for, does not start on the GPU cluster's environments
  (`import mujoco_py` fails in both); that is recorded as a gap, not tested
  here.

---

## Reproducing

```bash
OMP_NUM_THREADS=1 bash run.sh   # six cells in two waves; about 21 minutes on 24 cores
python summarise.py             # P1 cell by cell, P2, the figures above; writes summary.tsv
```

`summary.tsv` has one row per cell and seed and is what a coverage figure
should be drawn from. The first attempt, stopped on the Windows machine
before any cell finished, is in `results/attempt-1/` and is not read.
