# E36: over more iterations FPO++ runs pi0.5 into the ground, and DPPO barely moves it

2026-09-27/28 · qz103, two cards per run · pi0.5 on LIBERO-10 task 8 ·
protocol: [`PROTOCOL.md`](PROTOCOL.md) (`bb9d36b`, before any run)

---

## The result

Fifty-episode evaluations, initial states 0-49 (the untrained policy: 28-37
over seven evaluations, mean 30.9, sd 3.6):

| run | iteration 5 | iteration 10 |
| --- | --- | --- |
| `fpopp-s7` (FPO with FPO++'s chunk loss and per-sample ratio) | **5** / 50 | not run: no checkpoint |
| `fpopp-s8` | **0** / 50 | not run: no checkpoint |
| `dppo` (the `libero` variant) | 28 / 50 | **20** / 50 |

* **V1 and V2 pass.** Every run's flags took (`results/configs.txt`). Every
  evaluation ran fifty episodes, its client exited 0, and it is valid
  (`results/stageC_eval_correctness.tsv`).
* **P1 fails for both FPO++ runs**, and the fault is mine. The training
  harness, derived from E32's, wraps the clients in E32's `timeout 43200`.
  That fits two iterations. Ten FPO iterations of about 80 minutes do not
  fit. At 12 hours the clients were killed and the server with them, in the
  middle of the tenth learn step (`results/run.out`; the server log's
  "Received SIGTERM ... the model lock was still held"). So nine iterations
  learned and no iteration-10 checkpoint exists. DPPO's ten iterations of
  about an hour fit, and it ran to completion.
* **P2 (FPO++ holds at iteration 10) cannot be read as registered.** Its
  checkpoint does not exist. The evidence it would have read points one way:
  - At iteration 5 both runs are at the collapse line or below it (5 and 0).
  - The training rollouts had fallen to zero: 0.05, 0, 0 and 0, 0, 0 over
    iterations 7-9.
  - The one restart the protocol allows was not used. It would cost about
    thirteen GPU-hours to read a number the rest already gives.
* **P3 (DPPO holds at iteration 10) holds, on the line**: 20 of 50, where
  holding is 20 or more. That is also three standard deviations below the
  untrained policy's mean. It went 28 at iteration 5, then 20.

Neither algorithm made pi0.5 better. The status rule's "learns" is 42 of 50
at iteration 10, and nothing came near it. `pi0-policy` · DPPO stays
"holds", now after ten iterations. `pi0-policy` · FPO held for one update
(E32) and collapsed within five.

---

## What the logs show

`results/learn-stats.txt`, per iteration:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s7 training success | 0.60 | 0.38 | 0.73 | 0.40 | 0.10 | 0.14 | 0.05 | 0 | 0 |
| s7 CFM loss of its own samples | 0.11 | 0.13 | 0.16 | 0.26 | 0.81 | 5.8 | 28 | 74 | 114 |
| s7 clipped fraction | 0 | 0.46 | 0.52 | 0.46 | 0.43 | 0.50 | 0.58 | 0.75 | 0.96 |
| s8 training success | 0.67 | 0.50 | 0.47 | 0.18 | 0.13 | 0 | 0 | 0 | 0 |
| s8 CFM loss of its own samples | 0.12 | 0.12 | 0.19 | 1.9 | 10 | 52 | 180 | 338 | 407 |

The flow-matching loss of the policy's own freshly sampled actions is the
loss the next ratio starts from. It rose a thousandfold, and the success
fell with it. The velocity field stopped describing the distribution it
samples: the policy came apart, it did not merely drift. The clipped
fraction climbing towards one says that by the end almost every sample was
outside the trust region the moment the update began.

DPPO's `approx_kl` stayed between 1.3 and 4.1 x 10⁻⁷ for all ten
iterations, and the action expert moved 0.16% by iteration 5 and 0.24% by
iteration 10 (FPO++: 2.8% by iteration 5; `results/movement.txt`). DPPO kept
pi0.5 because it hardly touched it, as in E25. Where it did end up, 20 of
50, is below where it started.

---

## Where this points

The same shape as E37 on square: FPO's first update is harmless, later ones
degrade a pretrained policy. E37's critic never fit its returns. Here the
flow model's loss on its own samples runs away. Both ran with the parts of
FPO's own defaults that FPO++ does not share (`results/configs.txt`):
- rewards scaled by 10
- values and advantages recomputed every epoch
- one learning rate for actor and critic
- no gradient clipping

E39 is running FPO++'s
square fine-tuning in full (#91) to see whether the rest of FPO++'s setting
stops the square fall. If it does, the same settings go to pi0.5 next. For
DPPO on pi0.5 the question is the other one, why it barely moves, and
fpo-policy's history (E29, E30: the noise level) is the first place to look.

Checkpoints and tensorboards stay on qz103. The results, logs and harness
outputs are here.
