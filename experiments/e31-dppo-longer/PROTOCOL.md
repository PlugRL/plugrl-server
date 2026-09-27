# E31 measurement protocol (pre-registered)

**Written 2026-09-27, before any E31 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E28 ran `dppo-policy` under DPPO on Hopper and Walker2d for 100 iterations
(409,600 steps). Both rose on every seed and neither reached the bar of 500:
Hopper ended at 339, 430 and 276 and was still climbing; Walker2d ended at
260, 292 and 274 and had been flat since about iteration 50. **Run long
enough, does either learn its task?**

Same cells, same settings, three times as long: 300 iterations, 1,228,800
environment steps per seed - about the length of the standard million-step
MuJoCo benchmark, chosen before any data.

### What this cannot settle

* Why either stops where it does. Nothing but the length changes.
* Whether a different setting would do better. None is tried; the bar does
  not move.

---

## Declared in advance: what was already known

1. **E28's two cells** (`experiments/e28-dppo-learns/`), mean return per ten
   iterations, seeds 0 / 1 / 2:

   | cell | 11-20 | 41-50 | 91-100 |
   | --- | --- | --- | --- |
   | `dppo-hopper` | 77 / 74 / 80 | 195 / 214 / 201 | 339 / 430 / 276 |
   | `dppo-walker` | 186 / 172 / 155 | 251 / 224 / 267 | 260 / 292 / 274 |

   The best single Hopper iteration was 493 (seed 1).
2. **Learning is not reproducible bit for bit** once the first update has
   run (E20). E31's first 100 iterations are E28's settings on the same
   machine, not E28's numbers.

---

## Design

| cell | policy | algorithm | task | iterations | buffer / batch | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| `dppo-hopper` | `dppo-policy hopper` | `dppo hopper` | Hopper-v5 | 300 | 1,024 / 512 | 0, 1, 2 |
| `dppo-walker` | `dppo-policy walker` | `dppo walker` | Walker2d-v5 | 300 | 1,024 / 512 | 0, 1, 2 |

Both run through E24's `run_cell.sh`, unchanged, exactly as E28 ran them:
chunks of 4, so 4,096 environment steps per iteration; the variants'
10-iteration warm-up and then a constant learning rate; checkpoints every 20
iterations. Machine `guangzhao`, CPU, one thread per process. Code `main`,
client plugrl-env-client at 931ab56.

---

## The status rule

A cell **learns** if the mean return of iterations 291-300 is at least
**500** on at least **2 of 3** seeds - E23's, E24's and E28's bar for these
tasks. Otherwise it **did not learn in 1,228,800 steps**.

---

## Predictions, and what falsifies each

**P1 - both cells run end to end**: 300 iterations logged, the checkpoint at
300, no traceback, client exit 0, on every seed.

**P2 - `dppo-policy` learns Hopper.** `dppo-hopper` meets the status rule.

> Grounds: +62 to +66 by iterations 11-20, +184 to +202 by 41-50, +260 to
> +417 by 91-100, and still rising. Falsified if fewer than two seeds reach
> 500.

**P3 - `dppo-policy` learns Walker2d.** `dppo-walker` meets the status rule.

> What I expect: falsified. Between iterations 41-50 and 91-100 Walker2d
> gained 9, 69 and 7; at that pace it ends between about 300 and 560, most
> seeds below 500. Registered so that either outcome reads.

**Reported, not predicted:** mean return over iterations 91-100 (E28's
length), 191-200 and 291-300 for every seed; wall clock.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, the status rule, P2, P3, then the reported figures.
