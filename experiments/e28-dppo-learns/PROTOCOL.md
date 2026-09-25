# E28 measurement protocol (pre-registered)

**Written 2026-09-25, before any E28 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E24 showed that every DPPO combination on MuJoCo runs, and registered no
learning claim for any of them: DPPO from a random initialisation had not
learned HalfCheetah with `fpo-policy` (E17, E18). But in E24's twenty
iterations, unregistered, `dppo-policy` under DPPO rose on every seed of
Hopper and Walker2d, while `fpo-policy` under the same algorithm did not
move. Twenty iterations cannot call that learning.

**Run long enough to decide it: which DPPO cells learn their task, and is
it the policy class that decides?** Same five cells, same settings, five
times as long - the length at which E24 read FPO on Walker2d.

### What this cannot settle

* *Why* one policy class learns and the other does not, if that is what
  happens. `dppo-policy` and `fpo-policy` differ in the network, the action
  chunk (4 against 1), the number of denoising steps and how each step's
  noise is set. E28 can locate the difference in the policy class, not
  inside it.
* Anything about DPPO as published: DPPO fine-tunes policies pretrained on
  demonstrations, and every cell here starts from a random initialisation.

---

## Declared in advance: what was already known

1. **E24, the same five cells at twenty iterations**, on the same machine,
   code and settings. Return at the first iteration and the mean of the
   last ten, seeds 0, 1, 2:

   | cell | first | mean of iterations 11-20 | episode length |
   | --- | --- | --- | --- |
   | `dppo-hopper` | 11.2, 12.4, 15.8 | 75.7, 77.2, 81.5 | ~19 → ~50 |
   | `dppo-walker` | −2.2, −1.6, −2.6 | 220.0, 191.9, 156.5 | ~17 → ~194 |
   | `dppo-cheetah` | −386.3, −445.0, −382.1 | −384.0, −365.7, −418.2 | 1,000 |
   | `fpodppo-hopper` | 14.9, 19.7, 17.0 | 15.7, 17.6, 18.1 | ~21 → ~22 |
   | `fpodppo-walker` | 1.6, −0.0, 0.5 | 1.4, 0.1, 1.1 | ~20 → ~20 |

2. **`fpo-policy · DPPO · HalfCheetah` at this length** (E17, E18): 100
   iterations of 4,096 steps, three seeds, no learning - E18, with the
   observation statistics fixed, gained +5, +20, +2 against a threshold of
   +200.
3. **FPO at this length** learns Hopper (E23: 1,587, 2,006, 1,776) and
   Walker2d (E24: 964, 1,027, 1,101), from untrained policies near 0 to 20.
4. **Learning is not reproducible bit for bit** once the first update has
   run (E20), so E28's first twenty iterations are not expected to repeat
   E24's exactly; the collection before the first update is.

---

## Design

| cell | policy | algorithm | task | iterations | buffer / batch | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| `dppo-hopper` | `dppo-policy hopper` | `dppo hopper` | Hopper-v5 | 100 | 1,024 / 512 | 0, 1, 2 |
| `dppo-walker` | `dppo-policy walker` | `dppo walker` | Walker2d-v5 | 100 | 1,024 / 512 | 0, 1, 2 |
| `dppo-cheetah` | `dppo-policy cheetah` | `dppo cheetah` | HalfCheetah-v5 | 100 | 1,024 / 512 | 0, 1, 2 |
| `fpodppo-hopper` | `fpo-policy` | `dppo hopper` | Hopper-v5 | 100 | 4,096 / 2,048 | 0, 1, 2 |
| `fpodppo-walker` | `fpo-policy` | `dppo walker` | Walker2d-v5 | 100 | 4,096 / 2,048 | 0, 1, 2 |

Every cell is run by E24's own `run_cell.sh`, unchanged; only the number of
iterations differs from E24. Every iteration is 4,096 environment steps
(`dppo-policy` executes chunks of 4, so its buffer holds 1,024 entries), so
each seed sees 409,600 steps - E23's and E24's length for FPO. The DPPO
MuJoCo variants warm the learning rate up over 10 iterations and then hold
it (their scheduler's minimum equals its maximum), so the length changes
nothing about the schedule. Checkpoints every 20 iterations.

Machine: `guangzhao`, CPU, every process capped at one thread, as E24 and
E27. Code: E24's branch, whose code is `main` at 8812b54, run from its own
worktree through `PYTHONPATH`. Client: plugrl-env-client at 931ab56 (#9,
which changes only robomimic). The three `dppo-policy` cells run first, the
two `fpo-policy` cells after them.

---

## The status rule - what "learns" means, for every cell

A cell **learns** if, on at least **2 of 3** seeds:

* **Hopper, Walker2d**: the mean return of iterations 91-100 is at least
  **500** - E23's and E24's threshold for FPO on the same tasks, from
  untrained policies scoring −3 to 20 (known items 1, 3).
* **HalfCheetah**: the mean return of iterations 91-100 exceeds the first
  iteration's by at least **+200** - E18's threshold for the same question on
  the same task. An untrained policy scores −445 to −382 and runs every
  episode to 1,000 steps, so an absolute threshold would read the starting
  point, not the change; in E24's twenty iterations the change was −36 to
  +79.

Otherwise the cell **did not learn in 409,600 steps** - the matrix says
that, not "does not learn".

---

## Predictions, and what falsifies each

**P1 - every cell runs end to end.** For each cell, on every seed: all 100
iterations logged, the checkpoint at iteration 100, no traceback in the
server log, and a client that exits 0.

> Reported cell by cell, as in E24. Falsified, for that cell, by any seed
> failing any part.

**P2 - `dppo-policy` learns Walker2d.** `dppo-walker` meets the status rule.

> Grounds: +159 to +222 in twenty iterations on every seed, with episodes
> ten times longer. Falsified if fewer than two seeds reach 500: its early
> rise stalls.

**P3 - `dppo-policy` learns Hopper.** `dppo-hopper` meets the status rule.

> Weaker grounds: +64 to +66 in twenty iterations. Falsified if fewer than
> two seeds reach 500.

**P4 - `fpo-policy` under DPPO does not learn Hopper or Walker2d.** Neither
`fpodppo-hopper` nor `fpodppo-walker` meets the status rule.

> Grounds: no movement in twenty iterations (known item 1), and none in 100
> on HalfCheetah (known item 2). Falsified if either cell meets it: then
> `fpo-policy` learns under DPPO too, only more slowly, and E24's contrast was
> one of speed.

**Reading P2-P4 together.** If P2 or P3 holds and P4 holds, then under the
same algorithm, variant settings and data, whether DPPO learns from a random
start depends on the policy class, and E17 and E18's failure to learn belongs
to `fpo-policy`, not to DPPO.

**Reported, not predicted:** `dppo-cheetah`'s status; every cell's first and
last-ten returns and episode lengths; wall clock.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.
2. A failure caused by how a cell was launched rather than what it runs is
   fixed and the cell rerun once, recorded in `AMENDMENT.md`. A failure
   inside the code under test is a result and is not rerun.

---

## Reading order

P1 cell by cell, then P2, P3, P4, then `dppo-cheetah`'s status and the
descriptive figures.
