# E31: run three times as long, `dppo-policy` learns Hopper and Walker2d under DPPO

2026-09-27 · Linux workstation (`guangzhao`), CPU only · two DPPO cells,
three seeds each, 300 iterations of 4,096 steps · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`f8ffec0`, before any run)

---

## The result

E28 ran `dppo-policy` under DPPO on Hopper and Walker2d for 100 iterations;
both rose on every seed and neither reached the bar of 500. E31 ran the same
two cells, unchanged, to 300 iterations (1,228,800 steps per seed), with the
same bar.

| cell | mean of iterations 291-300, seeds 0 / 1 / 2 | status |
| --- | --- | --- |
| `dppo-policy` · DPPO · Hopper | **873 / 961 / 812**, from 11-16 | **learns** (3 of 3 past 500) |
| `dppo-policy` · DPPO · Walker2d | **528 / 622 / 585**, from −3 to −2 | **learns** (3 of 3 past 500) |

* **P1 holds, 2 of 2**: all six seeds logged 300 iterations, wrote the
  checkpoint at 300, have no traceback and a client that exited 0; 48 and 50
  minutes per cell.
* **P2 holds**: `dppo-policy` learns Hopper.
* **P3 holds**: `dppo-policy` learns Walker2d. I had written that I expected
  this one falsified; see below.

With E24, E27 and E28, the matrix of the two MLP policies now reads:

| | HalfCheetah | Hopper | Walker2d | robomimic square |
| --- | --- | --- | --- | --- |
| `fpo-policy` · FPO | learns (E6, E16) | learns (E23) | learns (E24) | runs (E27) |
| `fpo-policy` · DPPO | runs, no learning (E17, E18) | runs, no learning (E28) | runs, no learning (E28) | runs (E27) |
| `dppo-policy` · DPPO | learns (E28) | **learns (E31)** | **learns (E31)** | runs (E27) |

---

## The curves

Mean return over windows of ten iterations (the first: iteration 1 alone),
and each seed's best single iteration:

| cell, seed | 1 | 11-20 | 41-50 | 91-100 | 141-150 | 191-200 | 241-250 | 291-300 | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hopper, 0 | 11 | 76 | 189 | 336 | 490 | 685 | 629 | 873 | 1,073 (298) |
| Hopper, 1 | 12 | 73 | 188 | 405 | 707 | 831 | 1,005 | 961 | 1,336 (249) |
| Hopper, 2 | 16 | 80 | 207 | 350 | 609 | 753 | 731 | 812 | 1,122 (189) |
| Walker2d, 0 | −2 | 172 | 354 | 448 | 499 | 602 | 530 | 528 | 996 (186) |
| Walker2d, 1 | −2 | 172 | 243 | 436 | 591 | 540 | 671 | 622 | 983 (140) |
| Walker2d, 2 | −3 | 152 | 230 | 393 | 534 | 477 | 547 | 585 | 946 (240) |

Hopper climbs through all 300 iterations; by 191-200 every seed is past 500,
and episodes at the end last 456 to 652 steps, from 17 to 21. Walker2d's
ten-iteration mean is past 500 by 141-150 on seeds 1 and 2 and by 191-200 on
seed 0, and from there the windows above stay between 477 and 671: seed 0
ended below its own 191-200 mean. Its end, 528 to 622, is a pass, not a
plateau far above the bar.

---

## Where my expectation went wrong

For Walker2d I had written "falsified": between iterations 41-50 and 91-100
E28 gained 9, 69 and 7, and at that pace it would have ended between about
300 and 560. E31 did not repeat E28's first hundred iterations. At the same
point, iterations 91-100, E31's Walker2d stood at 448 / 436 / 393 where
E28's stood at 260 / 292 / 274 - higher on every seed, by 101 to 188.
Hopper, by contrast, came out close to E28 at that point (336 / 405 / 350
against 339 / 430 / 276).

Nothing in the code tells the two runs apart. Between E28's `d1f4783` and
E31's `f8ffec0` the DPPO path changed twice, and neither changes a number:
the log-probability clamp became configurable with its old bounds as the
default (#64), and a storage read wraps its result in `np.asarray`, which
returns the same array for every index DPPO uses (#58). Every seed's first
iteration matches E28's, and after the first update runs are not
reproducible bit for bit (E20, declared as known item 2). What E28 read as
Walker2d flattening after about fifty iterations was one run's curve, not a
property of the cell: E28's 100 iterations and E31's first 100 are two draws
of the same thing, and they differ by more than E28's gain over its last
fifty.

---

## What this does not show

* That 300 iterations is what either task needs. Both were past 500 in the
  ten-iteration windows by 191-200 on every seed; the protocol fixed 300
  before any data, and nothing shorter was read.
* How the cells compare with DPPO's published MuJoCo results. The bar is
  this project's, set in E23 for these tasks, and 1.2 million steps is
  about the length of the standard benchmark, not its protocol.
* Anything about `fpo-policy` under DPPO, which E30 is testing.

---

## Files

* `results/run.out` - the launcher's log: code commit, client commit, end
  time and the failed-seed count per cell (0 and 0).
* `results/<cell>.out`, `results/<cell>/{server,client}-seed<N>.log` - each
  cell's launcher output and each seed's server and client log.
* `results/verdicts.txt` - `summarise.py`'s output, run on `guangzhao`
  against the tensorboards and checkpoints, which stay there
  (`~/zuogou/plugrl/e31/experiments/e31-dppo-longer/results/<cell>/dppo/`).
* `summary.tsv` - first and last-ten return and episode length per seed.
