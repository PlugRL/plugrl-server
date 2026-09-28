# E37: from the same behaviour-cloned start, `fpo-policy` learns square under DPPO and loses ground under our FPO

2026-09-27 · Linux workstation (`guangzhao`), CPU only · two cells, three
seeds, three client processes per seed · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`fb93a91`, after the pilots in
`results/pilot.txt`, before the registered run)

---

## The result

Both cells started from E35's clone of `fpo-policy` on DPPO's 300
demonstrations, with the demonstrations' observation statistics frozen
(#82). Training-rollout success, start against the mean of the last ten
iterations:

| cell | seed | start | end | change |
| --- | --- | --- | --- | --- |
| `dppo-square` (40 iterations) | 0 | 0.390 | 0.781 | **+0.392** |
| | 1 | 0.411 | 0.784 | **+0.372** |
| | 2 | 0.372 | 0.798 | **+0.426** |
| `fpo-square` (60 iterations) | 0 | 0.517 | 0.256 | −0.261 |
| | 1 | 0.531 | 0.353 | −0.179 |
| | 2 | 0.550 | 0.392 | −0.158 |

(The starts differ because each algorithm samples the clone its own way.
DPPO adds noise of level 1.0 at every flow step. FPO integrates the flow
from random initial noise and adds nothing along the way.)

These are training-rollout rates. So, after the run and not registered,
every final checkpoint and the clone were evaluated for fifty episodes each
with E35's `eval_bc.sh`. That script integrates the flow's ODE with no noise
along the way, ends episodes on success, and uses 10 flow steps for FPO's
checkpoints and 20 for DPPO's:

| | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| the clone, unchanged | 0.50 | | |
| `fpo-square`, final | 0.24 | 0.30 | 0.42 |
| `dppo-square`, final | 0.86 | 0.78 | 0.90 |

The fall under FPO holds under evaluation, and the rise under DPPO is larger
there than in the training rollouts. The coverage figure's clip of the
median FPO seed succeeded 4 of 5; five episodes were luck against these
fifty.

* **P1 holds**: both cells logged every iteration on every seed, with no
  traceback and clients that exited 0. FPO took 5 hours and DPPO 9; both ran
  together from 12:30.
* **V1 passes**: every server restored the clone with `except-critic`, ran
  its cell's settings and kept the statistics frozen.
* **P3 holds**: `fpo-policy` learns square under DPPO, 3 of 3 seeds past
  +0.2.
* **P2 is falsified**: under FPO with FPO++'s square fine-tuning, as far as
  this FPO has it, it learns on 0 of 3 seeds. All three fell.

`fpo-policy` · DPPO · square moves from "runs" to "learns".
`fpo-policy` · FPO · square stays "runs": it runs end to end from a start
that succeeds half the time, and loses ground.

---

## The curves

Success by window of ten iterations:

| cell | seed | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51-60 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dppo-square` | 0 | 0.457 | 0.600 | 0.699 | 0.781 | | |
| | 1 | 0.415 | 0.603 | 0.724 | 0.784 | | |
| | 2 | 0.472 | 0.630 | 0.724 | 0.798 | | |
| `fpo-square` | 0 | 0.518 | 0.486 | 0.424 | 0.397 | 0.318 | 0.256 |
| | 1 | 0.507 | 0.500 | 0.505 | 0.492 | 0.395 | 0.353 |
| | 2 | 0.548 | 0.543 | 0.512 | 0.443 | 0.406 | 0.392 |

DPPO was still rising at 40. FPO did not crash: it held for 20 to 40
iterations and then drifted down on every seed. E36's FPO++ arms on pi0.5
fall the same way over their first seven iterations. The first update is
harmless (E32), and what follows is not.

---

## Where the two arms differ, read from their logs (not registered)

The critics. DPPO's scales rewards by the return's running deviation and
learns at 1e-3. Its explained variance rose from about 0.40 to about 0.61
on every seed. FPO's sees rewards multiplied by 10 (FPO's `reward_scaling`)
with no normalisation, and learns at the actor's rate, 1e-5. Its value loss
fell only from about 9 x 10⁴ to about 4 x 10⁴ over sixty iterations, while
the raw advantages kept a standard deviation of about 400. FPO logs no
explained variance, so how much of the return its critic explains is not
measured here. But nothing in these logs says it learned the return, and a
critic that has not learned turns the advantages into noise. On that noise
the policy took heavily clipped steps: 43% of samples at the first update
(the pilot), 42-50% over the last ten.

PROTOCOL.md declared this among the ways the FPO arm is not FPO++'s: FPO++
gives the critic ten times the actor's learning rate, 1e-4 against 1e-5.
The same list has squared error rather than Huber, no gradient clipping,
advantages normalised over the whole buffer, no clip on the old loss, and 3
client processes rather than 30 environments.

E37 also ran on code with the GAE episode-boundary defect (#86). Here it
touches only the first and last step of each 400-step episode, since
episodes do not end on success.

---

## What E37 does not show

* **That FPO cannot fine-tune square.** By this project's rule a
  platform that makes an algorithm fall is showing our defect, not a finding
  about the algorithm. The critic is the first suspect, and FPO++'s own
  critic settings are the first thing to try. That is a new, registered
  experiment, not a reading of this one.
* **That DPPO reproduces its paper here.** These are training rollouts with
  noise level 1.0, and the policy is a clone, not DPPO's diffusion policy.

Checkpoints and tensorboards stay on `guangzhao`. The logs, `verdicts.txt`
and `summary.tsv` are here.
