# E27 measurement protocol (pre-registered)

**Written 2026-09-25, after the pilot in `results/pilot.txt` and before any
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

The env client registers a robomimic family, and `dppo-policy` ships a
configuration and normalisation for robomimic's square task. Nothing had run
it. On the GPU cluster the client's environments cannot even import it:
robomimic 0.3.0, which the client's `robomimic` extra pins, imports
`mujoco_py` unconditionally, and neither cluster environment has it. E12
measured what installing it costs, not whether it runs.

**Do the platform's MLP policies train end to end on robomimic?** Three
cells, the robomimic column of E24's matrix.

### The environment, and why it is this one

On a Linux workstation (`guangzhao`), in a client venv of its own:
robomimic **v0.4.0** from its GitHub tag (PyPI stops at 0.3.0; v0.4.0 no
longer imports `mujoco_py` and supports robosuite 1.2 onwards), **robosuite
1.4.1** and **mujoco 2.3.7** - the pairing the GPU cluster already found
necessary, and the one the client's shipped environment metadata was written
for. `egl-probe`, a robomimic dependency, needs CMake to build; it was
installed into the venv, not the system. Before this file was written, the
client's own code built `NutAssemblySquare` from `square-img`, reset it and
stepped it twenty times, and its four low-dimensional keys summed to the 23
values `dppo-policy`'s square config expects.

### What this cannot settle

* Learning. Every cell claims only to run end to end.
* Seeded environments. `robomimic-v1` refuses a seed - it cannot reach the
  randomness of the robosuite simulation underneath - so each cell's three
  seeds vary the policy's initialisation and not the environment.

---

## Declared in advance: what was already known

1. **The first pilot** (`results/pilot.txt`): all three cells ran one
   iteration and exited 0 - and finished **0 episodes in 4,096 steps**.
   robomimic's environments never end an episode themselves (the shipped
   metadata sets `ignore_done`); robomimic's own rollouts stop on success or
   at a per-task horizon, and the client did neither. A platform defect,
   fixed in plugrl-env-client#9 (931ab56) with tests written first: episodes
   now end on success and at robomimic's rollout horizon, 400 for this task.
   The same PR makes the `robomimic` extra installable.
2. **The second pilot**, on the fixed client: each cell ran one iteration,
   finished **10 episodes** in about 4,096 steps - the 400-step horizon - and
   exited 0.
3. **An error at exit that is not a failure.** Every client log ends with an
   `OpenGL.error.GLError` from `eglMakeCurrent`, raised in robosuite's
   `EGLGLContext.__del__` as the interpreter shuts down, after the server has
   ended the run. It is printed as "Exception ignored in", changes nothing
   the run did, and is not counted against P1, which reads the server's log
   and the client's exit code.

---

## Design

| cell | policy | algorithm | iterations | buffer / batch | seeds |
| --- | --- | --- | --- | --- | --- |
| `fpo-rm` | `fpo-policy`, 23 state values (#61) | `fpo` | 20 | 4,096 / FPO's default | 0, 1, 2 |
| `fpodppo-rm` | `fpo-policy`, 23 state values | `dppo` default | 20 | 4,096 / 256 | 0, 1, 2 |
| `dppo-rm` | `dppo-policy`, robomimic `square` config | `dppo` default | 20 | 1,024 / 256 | 0, 1, 2 |

Task `square-img` (NutAssemblySquare), one env per client, the agentview
image at 84x84 since no policy reads images. `fpo-policy` executes every
action; `dppo-policy` its chunks of 4, so its buffer is 1,024 entries for
4,096 environment steps, as in E24. DPPO's default variant has no
learning-rate scheduler, so twenty iterations need no warmup. Code: `main`
plus #61 for the server, plugrl-env-client at 931ab56 (#9) for the client.
Every process capped at one thread, as E24 on the same machine.

---

## Predictions, and what falsifies each

**P1 - every cell runs end to end.** For each cell, on every seed: all
twenty iterations logged, the checkpoint at iteration 20, no traceback in
the server log, and a client that exits 0.

> Reported cell by cell. A cell that fails is reported as not running, with
> its error, and not rerun until it passes. Falsified, for that cell, by any
> seed failing any part.

**Reported, not predicted:** each run's first and last-ten returns, episode
lengths and success rates.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.
2. A failure caused by how a cell was launched rather than what it runs is
   fixed and the cell rerun once, recorded in `AMENDMENT.md`.

---

## Reading order

P1 cell by cell, then the descriptive figures.
