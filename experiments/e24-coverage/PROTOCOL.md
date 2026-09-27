# E24 measurement protocol (pre-registered)

**Written 2026-09-25, after the pilots in `results/pilot.txt` and before any
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

PlugRL separates the policy, the algorithm that trains it and the
environment it trains in. How many of those combinations actually run on it,
end to end? On a CPU, before today, the answer was two: `fpo-policy` with FPO
on HalfCheetah (E6, E16) and on Hopper (E23), and `fpo-policy` with DPPO on
HalfCheetah (E17-E19). DPPO's own policy class, `dppo-policy`, had never run
here at all.

E24 fills the MuJoCo block of that matrix: every combination of the two
MLP policies and the two algorithms that the code allows, on HalfCheetah,
Hopper and Walker2d. **Which of them run end to end, and does FPO also learn
Walker2d?**

### What each cell claims, and what it does not

A cell claims one of two things, never more than its evidence:

* **runs end to end** - every seed completes its iterations, writes its
  checkpoint, has no traceback and a client that exits 0. Nothing about
  learning. DPPO from a random initialisation did not learn HalfCheetah in
  this configuration (E17, E18), so no DPPO cell here claims to learn.
* **learns** - only `fpo-policy · FPO · Walker2d`, against a threshold set
  from the pilot's untrained policy.

### What the code rules out

FPO requires a flow policy (`BasePolicyGradientFlowPolicy`), and
`dppo-policy` is a diffusion policy, so **`dppo-policy · FPO` does not exist**
and is not a cell. DPPO accepts both. That leaves three policy-algorithm
pairs, and with three tasks, nine cells, three of which are already filled.

---

## Declared in advance: what was already known

1. **Filled before E24**: `fpo-policy · FPO` on HalfCheetah (learns, E6, E16)
   and Hopper (learns, E23); `fpo-policy · DPPO` on HalfCheetah (runs, E17,
   E18; barely changes a trained policy, E19).
2. **The pilots** (`results/pilot.txt`). `fpo-policy · FPO · Walker2d`, one
   iteration on three seeds: first-iteration return **0.71, −0.28, 0.64**,
   episode length about 19, all three ending cleanly. The five DPPO cells,
   one seed each: the first attempt at one iteration was refused at startup
   by DPPO's scheduler check, correctly - its warmup is 10 iterations - and a
   second at 11 iterations was stopped once every cell had completed its
   first learning updates without an error.
3. **`dppo-policy` in this repository.** Each variant reads its network, its
   action chunk (4 actions) and its D4RL min-max normalisation from files
   bundled under `meta/dppo`, for `halfcheetah-medium-v2`, `hopper-medium-v2`
   and `walker2d-medium-v2`. The `dppo` package it builds on comes from the
   pinned fork in `pyproject.toml`, installed from a local clone of that
   commit; installing it added packages and changed none.

---

## Design

| cell | policy | algorithm | task | iterations | buffer / batch | seeds | claims |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fpo-walker` | `fpo-policy` | `fpo` | Walker2d-v5 | 100 | 4,096 / FPO's default | 0, 1, 2 | learns |
| `fpodppo-hopper` | `fpo-policy` | `dppo hopper` | Hopper-v5 | 20 | 4,096 / 2,048 | 0, 1, 2 | runs |
| `fpodppo-walker` | `fpo-policy` | `dppo walker` | Walker2d-v5 | 20 | 4,096 / 2,048 | 0, 1, 2 | runs |
| `dppo-cheetah` | `dppo-policy cheetah` | `dppo cheetah` | HalfCheetah-v5 | 20 | 1,024 / 512 | 0, 1, 2 | runs |
| `dppo-hopper` | `dppo-policy hopper` | `dppo hopper` | Hopper-v5 | 20 | 1,024 / 512 | 0, 1, 2 | runs |
| `dppo-walker` | `dppo-policy walker` | `dppo walker` | Walker2d-v5 | 20 | 1,024 / 512 | 0, 1, 2 | runs |

`run_cell.sh` runs one cell; `run.sh` runs all six. Everything not in the
table is E17-E23's: CPU, one env per client, server and client seeds matched,
checkpoints every 20 iterations. `fpo-policy` executes every action it
returns (`replan-steps 1`, a chunk of 1); `dppo-policy` executes its whole
chunk of 4 before the next inference, as DPPO does, so one buffer entry is 4
environment steps. Its buffer is therefore 1,024 entries, so that an
iteration is still 4,096 environment steps, and its minibatch 512, so that
an epoch is still two minibatches, as with 4,096 and 2,048. 20 iterations
stay past the scheduler's 10-iteration warmup.

Scheduling: `fpo-walker` runs alongside `fpodppo-hopper` and
`fpodppo-walker` - nine server-client pairs - and the three `dppo-policy`
cells follow. No wall-clock prediction is registered, so contention has
nothing to invalidate.

---

## Predictions, and what falsifies each

**P1 - every cell runs end to end.** For each of the six cells, on every
seed: all its iterations logged, a checkpoint at its last save, no traceback
in the server log, and a client that exits 0.

> Reported cell by cell. A cell that fails is reported as not running, with
> its error, and the matrix shows it that way - not rerun until it passes.
> Falsified, for that cell, by any seed failing any part.

**P2 - FPO learns Walker2d.** In `fpo-walker`, the mean of iterations 91-100
is at least **500** on at least **2 of 3** seeds.

> The untrained policy scores −0.28 to 0.71 (known item 2), so 500 is far
> above anything a policy that learned nothing reaches. E23 used the same
> threshold on Hopper and every seed cleared it by more than 1,000. Falsified
> by two or more seeds below 500, and then, by the project's rule, it is a
> platform defect to locate before anything else.

**Reported, not predicted:** each cell's first and last-ten returns and
episode lengths, and each run's wall clock.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment -
   including a restart of the machine - recorded in `AMENDMENT.md`.
2. If a cell fails because of how it was launched rather than what it runs -
   a wrong flag, a port in use - the launch is fixed and the cell rerun once,
   recorded in `AMENDMENT.md`. A failure inside the code under test is a
   result and is not rerun.

---

## Reading order

P1 cell by cell, then P2, then the descriptive figures.
