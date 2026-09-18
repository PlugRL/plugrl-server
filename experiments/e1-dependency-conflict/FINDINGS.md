# E1: the claim that these dependencies cannot coexist is false

2026-09-09 · uv 0.9.8 (Windows) / 0.9.9 (WSL) · Python 3.11 · Linux side is
WSL2 Ubuntu 22.04

## In one sentence

The claim this experiment was written to test - *a training stack and an
environment stack cannot be installed in one Python environment* - is
**disproved by three rounds** of measurement, at the metadata layer, at the
install layer on Linux, and across environment families. A fourth round
measures the one difference that survives, and finds it holds for some
environments and not others.

This is kept in full because a disproved hypothesis is a result, and because
anyone building an argument on that claim should watch it fail first.

## Rounds 1-3: the claim fails

### Round 1 - training stack x environment, metadata layer (`run.sh`)

All 8 baselines resolved. Of 3 pairings expected to conflict, 1 failed.

| Pairing | Result |
|---|---|
| d4rl x reinflow | resolved - no conflict |
| robomimic x dppo | resolved - no conflict |
| **libero x openpi** | **failed** |

The single failure has one cause: **lerobot pins `gymnasium==0.29.1` while
the env client requires `>=1.2.0`**. That is an ordinary upstream pin in a
policy-side package, fixable with a pull request to lerobot - which is
exactly what makes it not structural.

### Round 2 - training stack x environment, install layer (`run-install.sh`, Linux)

| Case | Result |
|---|---|
| solo-training | installed (101s, 92 packages) |
| solo-envclient | installed (3s, 30 packages) |
| solo-env-d4rl | installed (20s, 69 packages) |
| **union: d4rl x reinflow** | **installed (144 packages)** |

The decisive one: a single `uv pip install` produced 144 packages containing

```
torch==2.7.1  d4rl==1.1  mujoco-py==2.1.2.14  cython==0.29.37
gymnasium==1.3.0  numpy==2.3.2  reinflow==1.0.0
```

**A modern training stack and an old simulator stack carrying `mujoco-py` and
`cython<3` do install together.**

The robomimic case failed twice before succeeding, and both failures were
defects in this harness rather than evidence:

1. `CMake must be installed` - a missing system tool.
2. After installing cmake 4.4.3, `egl-probe` failed to build.
3. After adding the repository's own
   `[tool.uv] extra-build-variables = { "egl-probe" = { CMAKE_POLICY_VERSION_MINIMUM = "3.5" } }`,
   both cases installed.

| Case | Result |
|---|---|
| solo-env-robomimic-fixed | installed (135s, 111 packages) |
| **union: robomimic x dppo (fixed)** | **installed (150 packages)** |

That union holds `torch==2.7.1`, `robomimic==0.3.0`, `robosuite==1.4.1`,
`mujoco-py==2.1.2.14`, `cython==0.29.37`, `d4rl==1.1`, `dppo==0.8.0` and
`egl-probe==1.0.2` at once.

**The install layer supports the claim 0 times out of 2.**

### Round 3 - environment x environment, metadata layer (`run-env-matrix.sh`)

This is the situation an evaluation matrix is actually in: several
environment families standing up at the same time.

| Case | Result |
|---|---|
| each of 5 families alone | all resolved |
| 5 cross-family pairs | all resolved |
| **all five families at once** | **resolved** |

0 out of 6.

### Tally

| How the question was asked | Supports the claim |
|---|---|
| training stack x environment (metadata) | 1/3 |
| training stack x environment (install, Linux) | **0/2** |
| environment x environment (metadata) | 0/6 |

## Round 4: deployment footprint (`run-footprint.sh`, measured on Linux)

With the coexistence claim gone, the difference that remains is not whether
the two sides *can* share an environment but what each machine producing
rollouts has to carry, and whether it needs a GPU. That decides whether a lab
workstation, a CI runner or a robot's onboard computer can be a rollout
source. Measured with uv 0.9.9 on WSL2 Ubuntu 22.04:

| Case | Role | Packages | Disk | CUDA |
|---|---|---|---|---|
| envclient-bare | rollout | 30 | **224M** | no |
| envclient-d4rl | rollout | 69 | **1008M** | no |
| envclient-robomimic | rollout | 111 | **7.2G** | **yes (16 wheels)** |
| monolith-d4rl | both | 142 | 6.5G | yes |
| monolith-robomimic | both | 150 | 7.4G | yes |
| training-only | trainer | 111 | 5.6G | yes |

### The footprint advantage depends on the environment package itself

| Environment | env client | monolith | advantage |
|---|---|---|---|
| dummy (`envclient-bare`) | 224M, no GPU | 6.5G, GPU | **29x, and no GPU** |
| d4rl | 1008M, no GPU | 6.5G, GPU | **6.4x, and no GPU** |
| **robomimic** | **7.2G, GPU** | 7.4G, GPU | **about none** |

**Correction, 2026-09-11.** That first row read "dummy / classic / atari"
until now, and only the dummy case was ever measured. `run-footprint.sh`
installs the env client's base dependencies and nothing else - the 30 packages
listed in `results-footprint/envclient-bare.log`, with no `ale-py`, no
`pygame`, no `shimmy`. That is exactly what the dummy environment needs, since
`dummy_env.py` sits in the base package, while `atari/` and `classic/` are
separate extras in `plugrl-env-client/pyproject.toml`. Round 3 resolved those
two extras (`results-matrix/single-atari.log`, `single-classic.log`) but never
installed them, so their size on disk is not measured anywhere in this
experiment. The 224M, and the 29x beside it, are the bare env client's
numbers; each extra adds to them by an amount not recorded here.

`envclient-robomimic` drags in `torch==2.14.0` and 16 nvidia wheels by
itself, because robomimic is a deep learning library that ships its own
policy learning code. For environments like it, "the environment side needs
no GPU" is simply not true.

**Overtaken, 2026-09-18, by [E12](../e12-cuda-free-rollout/FINDINGS.md).** The
sentence above is true of the install measured here and not of the
environment. On Linux `torch` ships a CUDA build whether or not CUDA is used;
pinning the CPU build of the same version takes `envclient-robomimic` to
**2.7G with no nvidia wheels**, and LIBERO - which this round never measured -
from 7.8G to **3.4G**, with `import robosuite` still working. E12 reproduced
this round's row exactly, 111 packages and 7.2G and 16 wheels on a different
machine, so the two sets of numbers are comparable. What survives of the
sentence is narrower: a **default** install of a robomimic-class environment
carries CUDA, and that is a packaging default rather than a requirement of the
environment.

**So "the env client has no torch" is a statement about
`plugrl-env-client`'s own dependencies, and does not generalise to every
environment family.** Any argument built on footprint has to be made per
environment, and has to name the counterexample.

## What this does and does not support

Supported:

* The two stacks install together, on Linux, including the combinations that
  were expected to be impossible.
* For the two rollout cases that were installed and weighed, the env client
  is smaller than a monolithic install and needs no GPU: 6.4x for d4rl, 29x
  for the bare client carrying no environment package at all. The other
  environment families were resolved but never installed, so the 29x should
  not be read as covering them.
* One real dependency conflict exists (lerobot's gymnasium pin) and it is an
  ordinary upstream pin.

Not supported:

* That the process boundary is *forced* by dependency conflicts. It is a
  choice, and has to be argued on its own merits.
* That an env client never needs a GPU or a deep learning stack. Robomimic is
  a counterexample measured here.

## Reproducing

```bash
bash run.sh                     # round 1, metadata layer
wsl bash wsl-bootstrap.sh       # round 2, Linux install layer
bash run-env-matrix.sh          # round 3, environment x environment
wsl bash wsl-footprint.sh       # round 4, deployment footprint
```

Logs are in `results/`, `results-install/`, `results-matrix/` and
`results-footprint/` respectively. Dependency SHAs are pinned at the top of
each script as resolved on 2026-09-09, so a rerun measures the same thing
rather than whatever the forks have become.
