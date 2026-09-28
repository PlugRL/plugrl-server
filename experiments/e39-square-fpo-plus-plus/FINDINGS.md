# E39: with the rest of FPO++'s fine-tuning, FPO stops pulling a cloned policy down on square and lifts it - short of the bar in 4.8M steps

2026-09-27/28 · Linux workstation (`guangzhao`), CPU only · one cell, three
seeds, 100 iterations of 48,000 steps · protocol: [`PROTOCOL.md`](PROTOCOL.md)
(`5c6ccd7`, after the pilot in `results/pilot.txt`, before the registered
run)

---

## The result

E35's clone of `fpo-policy`, under FPO with FPO++'s square fine-tuning in
full (#91): E37's policy loss plus FPO++'s optimizer, clipping, advantage
handling, Huber error and raw rewards. Training-rollout success:

| seed | start: iterations 1-2 | end: iterations 91-100 | gain |
| --- | --- | --- | --- |
| 0 | 0.495 | 0.787 | **+0.292** |
| 1 | 0.524 | 0.684 | +0.160 |
| 2 | 0.518 | 0.711 | +0.193 |

* **P1 holds**: 100 iterations on every seed, ten checkpoints each, no
  traceback, clients exited 0. It took 7.6 hours, beside E40.
* **V1 passes**: every server ran every setting `summarise.py` lists.
* **V2 passes**: the critic learns. Explained variance over iterations
  11-20 was 0.48 / 0.44 / 0.44, and 0.66-0.68 over the last ten.
* **P2 is falsified**: 1 of 3 seeds gained 0.2. The other two gained 0.16
  and 0.19. By the registered rule the cell does not learn in 4.8M steps,
  and `fpo-policy` · FPO · square stays "runs".

Fifty-episode evaluations after the run, with E35's harness and the value
head these checkpoints have. The harness integrates the flow's ODE and ends
an episode on success:

| | E37 (FPO++'s loss only) | E39 (FPO++'s fine-tuning) |
| --- | --- | --- |
| the clone | 0.50 | 0.50 |
| seed 0, final | 0.24 | **0.80** |
| seed 1, final | 0.30 | **0.64** |
| seed 2, final | 0.42 | **0.64** |

---

## What changed between E37 and E39

The policy fell in E37 and rose here, on every seed, in training and in
evaluation. The one thing E39 changed was FPO++'s settings beyond its loss:
- the critic's own learning rate, ten times the actor's
- AdamW
- gradient clipping
- per-minibatch advantages computed once an iteration
- raw rewards
- the Huber error
- episodes ending on success

The GAE fix (#86) came between the two runs as well. E37's critic never fit
the returns; E39's explains two thirds of them. Which of these changes did
the work is not separated: they were added together, as PROTOCOL.md says.
The fall E37 recorded was our defect, the platform running FPO without the
rest of the fine-tuning it was published with, and this removes it.

Success by window of ten iterations:

| seed | 1-10 | 11-20 | 21-30 | 31-40 | 41-50 | 51-60 | 61-70 | 71-80 | 81-90 | 91-100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.525 | 0.574 | 0.630 | 0.669 | 0.690 | 0.701 | 0.713 | 0.774 | 0.759 | 0.787 |
| 1 | 0.538 | 0.574 | 0.593 | 0.610 | 0.639 | 0.660 | 0.657 | 0.644 | 0.687 | 0.684 |
| 2 | 0.531 | 0.535 | 0.600 | 0.625 | 0.659 | 0.670 | 0.712 | 0.692 | 0.661 | 0.711 |

Every seed rose through the whole run, the last window near the best, and
none has levelled off. The clipped fraction stayed between 0.19 and 0.29
(E37: 0.42-0.50 at the end).

---

## What E39 does not show

* **That it learns by the bar.** Two seeds fell short of +0.2 at 4.8M steps.
  FPO++'s own square runs are 8M steps (about 167 of these iterations).
* **Which setting mattered**, as above.
* **Anything about pi0.5** (E36), where the same FPO++ loss ran with the same
  FPO defaults and collapsed.

Checkpoints and tensorboards stay on `guangzhao`. The logs, `verdicts.txt`
and `summary.tsv` are here.
