# E34: in DPPO's own fine-tuning setting, `dppo-policy` learns robomimic square on PlugRL

2026-09-27 · Linux workstation (`guangzhao`), CPU only · one cell, three
seeds, 40 iterations of 80,000 steps · protocol: [`PROTOCOL.md`](PROTOCOL.md)
(`360acaf`, after the pilot in `results/pilot.txt`, before the registered run)

---

## The result

E27 ran `dppo-policy` under DPPO on square from random weights and could only
say it runs: success 0 throughout. E34 ran it the way DPPO's authors do -
their released pretrained policy (its `ema` weights, #73), the last 10 of 20
denoising steps fine-tuned through a frozen copy of the first 10 (#76), and
every value of their square config (#77) - with the rule for "learns" fixed
in advance.

| seed | start: iterations 1-3 | end: iterations 31-40 | gain |
| --- | --- | --- | --- |
| 0 | 0.361 | 0.649 | **+0.288** |
| 1 | 0.361 | 0.675 | **+0.313** |
| 2 | 0.330 | 0.702 | **+0.372** |

(Training-rollout success, with DPPO's sampling noise; the start is the
released policy unchanged, since the first two iterations train only the
critic.)

* **P1 holds**: all three seeds logged 40 iterations, wrote their
  checkpoints, have no traceback and a client that exited 0; about 5.2 hours.
* **V1 passes**: every server logged `ft_denoising_steps=10` and loaded the
  checkpoint's `ema` weights.
* **P2 holds**: `dppo-policy` learns square - 3 of 3 seeds past +0.2.

`dppo-policy` · DPPO · square moves from "runs" to "learns".

---

## The curves

Mean success per window of ten iterations:

| seed | 1-10 | 11-20 | 21-30 | 31-40 | best (iteration) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.365 | 0.493 | 0.553 | 0.649 | 0.697 (38) |
| 1 | 0.372 | 0.506 | 0.609 | 0.675 | 0.737 (38) |
| 2 | 0.386 | 0.526 | 0.591 | 0.702 | 0.783 (38) |

Still rising at 40. The mean return - one per step spent succeeding, since
episodes do not end on success - went from 51-55 over the first three
iterations to 118-132 over the last ten. `approx_kl` averaged 3 to 9 x 10⁻⁴
over iterations 3-12 and about 1.6 x 10⁻⁴ over the last ten; `clipfrac`
17% and 13%.

DPPO's paper reports its deterministic evaluations rising from about 0.40 to
about 0.8 over the first 2.5 to 3 million steps; 40 iterations here are 3.2
million. The two are not the same measurement - these are the training
rollouts, with a deviation of at least 0.1 at every denoising step - and E34
does not claim to reproduce the paper's number, only that the setting learns
through PlugRL.

---

## What this does not show

* Deterministic success, or success after DPPO's full 201 iterations.
* Anything about `fpo-policy` on square (E37).

The release's single-iteration swings are larger than binomial noise at
about 200 episodes per iteration: seed 0's three iterations of the
unchanged policy ran 0.40, 0.38, 0.31. The rule compares means over three
and over ten iterations, and every seed cleared it by a margin larger than
those swings.

---

## Files

* `results/run.out`, `results/dppo-square.out`,
  `results/dppo-square/{server,client}-seed<N>.log` - the launcher, the
  cell and each seed's server and client log.
* `results/verdicts.txt` - `summarise.py`'s output, run on `guangzhao`
  against the tensorboards and checkpoints, which stay there.
* `results/pilot.txt` - the three-iteration pilot.
* `summary.tsv` - start, end and gain per seed.
