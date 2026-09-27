# E27: the platform's MLP policies run on robomimic

2026-09-25 · Linux workstation (`guangzhao`), CPU only · two policies, two
algorithms, robomimic's `square` task, three seeds per cell · protocol:
[`PROTOCOL.md`](PROTOCOL.md) · pilots: [`results/pilot.txt`](results/pilot.txt)

---

## The result

The env client registers a robomimic family and `dppo-policy` ships a
configuration and normalisation for robomimic's `square` task, but nothing
had ever run them: on the GPU cluster the client cannot even import
robomimic. E27 is the robomimic column of E24's matrix.

**All three cells run end to end, on every seed.**

| cell | policy · algorithm · task | claim | result |
| --- | --- | --- | --- |
| `fpo-rm` | `fpo-policy` · FPO · square | runs | runs, 3 of 3 (14 min) |
| `fpodppo-rm` | `fpo-policy` · DPPO · square | runs | runs, 3 of 3 (15 min) |
| `dppo-rm` | `dppo-policy` · DPPO · square | runs | runs, 3 of 3 (17 min) |

**P1 holds, 3 of 3**: on all nine seeds all twenty iterations are logged,
the checkpoint at iteration 20 is written, no server log holds a traceback
and every client exits 0. Each seed ran 81,920 environment steps as 204
episodes. The only error in any client log is the one declared in advance
(`PROTOCOL.md`, item 3): robosuite freeing its EGL context as the interpreter
shuts down, printed as "Exception ignored in", after the server has ended the
run.

With E24, the MLP policies' block of the matrix now reads:

| | HalfCheetah | Hopper | Walker2d | robomimic square |
| --- | --- | --- | --- | --- |
| `fpo-policy` · FPO | learns (E6, E16) | learns (E23) | learns (E24) | **runs (E27)** |
| `fpo-policy` · DPPO | runs (E17, E18) | runs (E24) | runs (E24) | **runs (E27)** |
| `dppo-policy` · DPPO | runs (E24) | runs (E24) | runs (E24) | **runs (E27)** |

---

## Getting it to run found three defects

Each was fixed, with tests written first, before the registered run:

1. **The `robomimic` extra could not be installed.** It pinned robomimic
   0.3.0, the last on PyPI, which imports `mujoco_py` unconditionally.
   Now robomimic v0.4.0 from its tag, robosuite 1.4.1 and mujoco 2.3.7
   (plugrl-env-client#9).
2. **No episode ever ended.** robomimic's environments never end an episode
   themselves and the client neither stopped on success nor at a horizon:
   the first pilot finished 0 episodes in 4,096 steps. Episodes now end on
   success and at robomimic's rollout horizon, 400 steps for `square`
   (plugrl-env-client#9).
3. **`fpo-policy` could not read robomimic's observation.** It read
   `states["obs"]` only; robomimic hands over named low-dimensional keys.
   `--policy.state-keys` now concatenates them in order and checks the width
   (#61).

---

## Nothing was learned, and nothing was expected to be

Every iteration of every seed has a success rate of 0, a return of 0 and
episodes of exactly 400 steps - the horizon. The shipped `square` metadata
sets `reward_shaping: false`, so the reward is 1 only on success. In 204
episodes per seed, a randomly initialised policy never assembled the nut
once, and a sparse reward that never fires gives an on-policy algorithm
nothing to follow.

No cell claimed to learn. `dppo-policy` can load a pretrained actor through
`checkpoint_path`, and DPPO's own robomimic results fine-tune a policy
pretrained on demonstrations; no such checkpoint was given here, so every
cell started from a random initialisation, as in E24.

---

## What this does and does not support

**Supported:**

* Both MLP policies run on robomimic's `square` task under DPPO, and
  `fpo-policy` runs under FPO - every combination of these policies and
  algorithms the code allows, on a fourth task family.
* robomimic now installs and runs through the client, in its own
  environment, on a Linux machine with EGL.

**Not supported:**

* Anything about learning on robomimic. That needs either a pretrained
  actor for `dppo-policy` or a shaped reward, and neither was registered.
* That robomimic runs on the GPU cluster: the fixed extra was installed only
  on `guangzhao`.
* Seeded environments. `robomimic-v1` refuses a seed, so the three seeds vary
  only the policy's initialisation.

---

## Reproducing

```bash
OMP_NUM_THREADS=1 bash run.sh   # three cells, nine runs at once; about 17 minutes on 24 cores
python summarise.py             # P1 cell by cell, then the figures above; writes summary.tsv
```

`results/verdicts.txt` is `summarise.py`'s output. `summary.tsv` has one row
per cell and seed in E24's columns, so a coverage figure can read both the
same way. Checkpoints and tensorboard files stay on `guangzhao`
(`.gitignore`).
