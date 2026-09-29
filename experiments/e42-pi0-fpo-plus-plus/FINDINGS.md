# E42: with FPO++'s fine-tuning in full, FPO no longer collapses pi0.5 - it holds on both seeds through ten iterations, and improves it on neither by the bar

2026-09-28/29 · qz103, two cards per run · pi0.5 on LIBERO-10 task 8 ·
protocol: [`PROTOCOL.md`](PROTOCOL.md) (`db079f0`, after the pilot in
`results/pilot.txt`, before the registered run)

---

## The result

E36's FPO++ arm (FPO++'s chunk loss and per-sample ratio) now runs with the
rest of FPO++'s fine-tuning, which E36 had left on FPO's defaults:
- the critic's own learning rate, 1e-4
- AdamW
- gradient clipping at 25
- advantages normalised per minibatch
- raw rewards
- truncation treated as termination
- the Huber error
- lambda 0.99
- clip 0.01

Two seeds ran ten iterations of 4,096 chunks. Each fifty-episode evaluation
covers initial states 0-49:

| run | iteration 5 | iteration 10 |
| --- | --- | --- |
| `fpopp-s7` | 39 / 50 | **40** / 50 |
| `fpopp-s8` | 31 / 50 | **27** / 50 |
| E36 with FPO's defaults, the same seeds | 5 and 0 | not reached |
| the untrained policy, seven evaluations | 28-37, mean 30.9 | |

* **P1 holds.** Both runs finished ten iterations with ten checkpoints each
  (4,100 to 40,960), no traceback and no out-of-memory, and every client
  exited 0. The 30-hour client timeout that replaced E36's 12 hours was
  enough.
* **P2 holds.** Both runs hold at iteration 10, at 40 and 27 against a line
  of 20.
* **All four evaluations are valid.** Each ran fifty episodes, and the
  server's episode and step counts match the clients' exactly
  (`results/stageC_eval_correctness.tsv`).

**Neither run learns by E36's bar**, 42 of 50 at iteration 10, which is what
PROTOCOL.md expected.
- `fpopp-s7` ends at 40, above every one of the untrained policy's seven
  evaluations: 2.5 of their standard deviations over their mean.
- `fpopp-s8` ends at 27, one below their lowest.

With two seeds that is one run possibly improving and one holding, not an
improvement.

---

## What changed E36's collapse

E36 and E42 differ only in the settings above. E36's runs went to 5 and 0 by
iteration 5. On the same seeds E42's are at 39 and 31, and at 40 and 27 five
iterations later.

This is the same finding E37 and E39 made on square with `fpo-policy`. FPO
run without the rest of the fine-tuning it was published with pulls a
pretrained policy down, and with it the fall goes away. **The collapse E11,
E14 and E36 recorded was the platform running FPO incompletely.** It was not
a property of FPO on pi0.5, as the project's rule says to assume until shown
otherwise. Which of the settings did the work is not separated: they were
added together.

How far the policy moved does not explain it:
- **E42's movement.** At iteration 10 both runs had moved every module group
  by about 1% relative distance, the action expert by 1.05% and 1.02%
  (`results/movement.txt`).
- **E15's.** Under FPO's defaults, E15 found that under 1% of movement in the
  action expert was enough to take pi0.5 to 0.
- **So:** the magnitude is the same, and E21's conclusion stands: the
  direction is what breaks it.

---

## The two runs are not alike

Training rollouts, per iteration (`results/learn-stats.txt`). The CFM loss
is `fpo/initial_cfm_loss_mean`, the same series E36's findings report:

| | 1 | 5 | 9 | 10 |
| --- | --- | --- | --- | --- |
| s7 training success | 0.63 | 0.64 | 0.80 | 0.71 |
| s8 training success | 0.60 | 0.61 | 0.55 | 0.47 |
| s7 CFM loss of its own samples | 0.107 | 0.125 | 0.119 | 0.137 |
| s8 CFM loss of its own samples | 0.107 | 0.125 | 0.194 | 0.223 |
| s7 clipped fraction | 0 | 0.34 | 0.34 | 0.32 |
| s8 clipped fraction | 0 | 0.32 | 0.38 | 0.43 |

`fpopp-s8` drifts the way E36's runs collapsed:
- its training success falls
- its clipped fraction rises
- the flow-matching loss of its own fresh samples doubles

It does all of this far more slowly than E36, whose clipped fraction reached
0.75-0.99 and whose loss reached 114 and 407. Whether s8 would collapse given
more iterations is open. `fpopp-s7` shows none of it.

---

## What E42 does not show

* **That RL improves pi0.5.** One run of two ends above the untrained range,
  and neither reaches the bar.
* **What happens past ten iterations**, which s8's drift makes the open
  question.
* **Which of FPO++'s settings matters**, since they came together.
* **Any other task than LIBERO-10 task 8.**

Checkpoints stay on qz103. The evaluation tables, learn-step statistics,
movement, the configurations and recorded environments of both processes,
and `run.out` are here.
