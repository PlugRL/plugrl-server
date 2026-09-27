# E33: with DPPO's structure, `fpo-policy` learns all three MuJoCo tasks under DPPO

2026-09-27 · Linux workstation (`guangzhao`), CPU only · three cells, three
seeds each, 300 iterations of 4,096 steps · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`25d6081`, before any run)

---

## The result

E30 found that `fpo-policy` under DPPO learns Hopper once it carries
`dppo-policy`'s structure - three hidden layers of 512, four actions per
inference, twenty flow steps - at noise level 1.0 (E30's `all`), and chose
that configuration after seeing Hopper. E33 fixed it unchanged and ran it on
all three tasks, Hopper on seeds E30 had not used.

| cell | mean of iterations 291-300, per seed | status |
| --- | --- | --- |
| `fpo-policy` · DPPO · HalfCheetah (seeds 0-2) | **+1,331 / +1,411 / +1,294** over the first iteration | **learns** (3 of 3 past +200) |
| `fpo-policy` · DPPO · Hopper (seeds 3-5) | **1,025 / 998 / 1,037** | **learns** (3 of 3 past 500) |
| `fpo-policy` · DPPO · Walker2d (seeds 0-2) | **562 / 283 / 569** | **learns** (2 of 3 past 500) |

* **P1 holds**: all nine seeds logged 300 iterations, have their
  checkpoints, no traceback and a client that exited 0; 40 to 42 minutes per
  cell, the three at once.
* **V1 passes**: every cell's logged entropy was -0.0040 at every
  iteration, as E30's `all`.
* **P2 holds**: Hopper, on seeds 3, 4, 5.
* **P3 holds**: Walker2d, on two seeds of three - seed 1 stayed at 283.
* **P4 holds**: HalfCheetah.

With E24, E27, E28 and E31, the matrix of the two MLP policies now reads:

| | HalfCheetah | Hopper | Walker2d | robomimic square |
| --- | --- | --- | --- | --- |
| `fpo-policy` · FPO | learns (E6, E16) | learns (E23) | learns (E24) | runs (E27) |
| `fpo-policy` · DPPO | **learns (E33)** | **learns (E33)** | **learns (E33)** | runs (E27) |
| `dppo-policy` · DPPO | learns (E28) | learns (E31) | learns (E31) | runs (E27) |

---

## The curves

Mean return over windows of ten iterations (the first: iteration 1 alone):

| cell, seed | 1 | 41-50 | 91-100 | 141-150 | 191-200 | 241-250 | 291-300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HalfCheetah, 0 | −186 | −261 | −200 | −140 | −119 | 362 | 1,145 |
| HalfCheetah, 1 | −226 | −268 | −199 | −125 | −65 | 438 | 1,185 |
| HalfCheetah, 2 | −271 | −235 | −170 | −123 | 318 | 855 | 1,022 |
| Hopper, 3 | 19 | 243 | 939 | 1,013 | 931 | 1,029 | 1,025 |
| Hopper, 4 | 16 | 248 | 668 | 1,003 | 1,008 | 992 | 998 |
| Hopper, 5 | 18 | 247 | 944 | 1,042 | 986 | 1,028 | 1,037 |
| Walker2d, 0 | 3 | 267 | 271 | 292 | 321 | 438 | 562 |
| Walker2d, 1 | 3 | 276 | 280 | 285 | 282 | 283 | 283 |
| Walker2d, 2 | 3 | 278 | 282 | 316 | 315 | 403 | 569 |

The three tasks learn on different clocks. Hopper is past 500 by
iterations 91-100 on every seed, as in E30, and flat at about 1,000 from
iteration 150. HalfCheetah loses ground for fifty iterations, is still
negative at 191-200 on two seeds, and climbs about 1,200 in the last
hundred. Walker2d sits near 280 from iteration 40 to 200 on every seed;
seeds 0 and 2 then climb and are still climbing at 300, seed 1 does not
move.

---

## What the Hopper and Walker2d numbers are

Both tasks pay 1 per step for staying up. Hopper's episodes here reach the
1,000-step limit - 968 to 1,000 steps over the last ten iterations - for a
return of about 1,000: 1.00 to 1.06 per step, so almost all of it is staying
up. Walker2d's two passing seeds make 1.09 and 1.02 per step over 514 and
558 steps. By the bar these cells have used since E23 (500), both learn; it
is a bar a policy can meet by staying up for 500 steps.

The other policies that met it did more on Hopper and a little more on
Walker2d. Per step, over the last ten iterations: `fpo-policy` under FPO on
Hopper 2.80 to 2.97 (E23); `dppo-policy` under DPPO 1.25 to 1.91 on Hopper
and 1.13 to 1.25 on Walker2d (E31). The coverage figure's clips will show
the difference.

HalfCheetah has no reward for staying up. Its +1,300 is running.

---

## What this does not show

* Which part of the structure matters on these tasks; E30 took them apart
  on Hopper only.
* Whether 300 iterations is enough for Walker2d: two seeds were still rising
  and one had not started.
* Anything about FPO's default structure under DPPO, which E28 already
  showed does not learn.

---

## Files

* `results/run.out` - the launcher's log: code, client, failed seeds per cell
  (all 0).
* `results/<cell>.out`, `results/<cell>/{server,client}-seed<N>.log` - each
  cell's launcher output and each seed's server and client log.
* `results/verdicts.txt` - `summarise.py`'s output, run on `guangzhao`
  against the tensorboards and checkpoints, which stay there
  (`~/zuogou/plugrl/e33/experiments/e33-fpo-dppo-structure/results/<cell>/dppo/`).
* `summary.tsv` - first and last-ten return and episode length per seed.
