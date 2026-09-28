# E32: FPO's first update no longer destroys pi0.5 once it scores an action chunk as FPO++ does

2026-09-27 · qz103, 8 x RTX 3090 · four training arms of two iterations each
and five fifty-episode evaluations, pi0.5 on LIBERO-10 task 8 · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`78ff313`, after the two harness checks in
`results/pilot.txt`, before any registered run)

---

## The result

Since E14, one FPO iteration has taken pi0.5 from about 30 of 50 to 0. E32
ran FPO's authors' revision, FPO++, against it, one change at a time: how a
chunk is scored (FPO++'s chunk loss) and how many ratios are formed (one per
Monte Carlo sample). Every arm first trained the critic for an iteration, as
FPO++ does, and then made one policy update.

| arm | chunk loss | ratio | fifty episodes after the update |
| --- | --- | --- | --- |
| `control` | FPO's | per action | **0** (0.000 - 0.071) |
| `persample` | FPO's | per sample | **0** (0.000 - 0.071) |
| `chunk` | FPO++'s | per action | **26** (0.385 - 0.652) |
| `fpopp` | FPO++'s | per sample | **33** (0.522 - 0.776) |
| `base` | untrained | | 36 (0.583 - 0.825) |

(95% Wilson intervals of the success rate.)

* **V1 passes**: every arm's iteration-1 checkpoint left all 201 expert
  tensors and the projections bit-identical to the base - the first
  iteration trained only the critic.
* **V2 passes**: each arm's config line shows its chunk loss, ratio, the
  warmup, learning rate 1e-5 and clip 0.05.
* **V3 passes**: five valid evaluations of fifty episodes.
* **P1 holds**: the collapse reproduces on this code, with a critic-only
  iteration first: 0 of 50.
* **P2 holds**: FPO++ prevents it, 33 of 50.
* **P3 holds**: its chunk loss alone does, 26 of 50.
* **P4 holds**: its per-sample ratio alone does not, 0 of 50.

---

## Not because the update was smaller

Every arm moved the action expert further than the smallest movement E15
found destructive (0.66%). Relative distance from the base after the update:

| group | `control` | `chunk` | `persample` | `fpopp` |
| --- | --- | --- | --- | --- |
| expert | 2.02% | 1.59% | 2.01% | 1.64% |
| mod | 2.66% | 2.37% | 2.56% | 2.38% |
| attn | 2.41% | 1.64% | 2.35% | 1.71% |
| mlp | 2.06% | 1.62% | 2.08% | 1.70% |
| io | 1.43% | 1.43% | 1.36% | 1.66% |

The two arms that kept the policy moved it about 80% as far as the two that
destroyed it. The difference is where the update went, which is what E21
and E22 had pointed at from the other side: the same FPO, scoring a chunk
differently, moves pi0.5 in a direction it survives.

---

## What the ratio saw

The policy update's statistics, means over its minibatches:

| arm | loss at collection | loss during the update | ratio | clipped |
| --- | --- | --- | --- | --- |
| `control` | 0.00051 | 0.082 | 0.925 | 51% |
| `persample` | 0.00057 | 0.069 | 0.939 | 41% |
| `chunk` | 0.130 | 0.223 | 0.920 | 56% |
| `fpopp` | 0.130 | 0.272 | 0.907 | 56% |

Under FPO's loss - the eps-weighted error averaged over all 320 elements of
a chunk, 285 of them padding or steps the client never executed - the
update raised the loss 120- to 160-fold. Under FPO++'s - the velocity error
on the 5 executed steps and the 7 dimensions LIBERO uses, summed over steps -
it raised it about twofold. The ratios were pushed down and clipped about as
often in all four arms; what the loss counted as a change differed.

---

## What this does not show

* Which part of FPO++'s chunk loss does it. It changes four things at once:
  velocity instead of eps supervision, continuous instead of grid times,
  only executed steps and used dimensions, and a sum over steps instead of a
  mean.
* That FPO++ improves pi0.5. 33 against the untrained 36 is within noise:
  the policy survived one update, and nothing here says more updates help.
* Anything beyond one task, one seed, one evaluation per arm.

The untrained policy scored 36, at the top of the range seen before (28 to
37 over seven evaluations of an unperturbed actor, E15; 29 in E26).

---

## For the coverage figure

`pi0-policy` · FPO moves from "collapses" to "holds": with FPO++'s chunk loss
one update leaves pi0.5 at 33 of 50. It has not been shown to learn.

---

## Files

* `results/stageC_eval.tsv`, `results/stageC_eval_correctness.tsv` - the
  evaluation harness's own tables.
* `results/movement-iteration1.txt`, `results/movement.txt` -
  `movement.py` against E15's base dump, after each iteration.
* `results/learn-stats.txt` - `tb_read.py` over each arm's tensorboard.
* `results/configs.txt` - each arm's config line from its server log.
* `results/verdicts.txt` - `summarise.py`'s output.
* `results/run.log`, `results/run.out`, `results/train-*.out`,
  `results/eval-*.out`, `results/server-*.log` - the launcher, each training
  and evaluation harness, and each training server's log.
* `results/environment-*.txt` - the evaluation harness's package and source
  manifests, the latter taken through the same `PYTHONPATH` as the server:
  `fpo_config.py` and `utils.py` hash to #74's blobs (5272832), which E25's
  and E26's manifests could not show.
* `results/pilot.txt`, `results/train-pilot*.out` - the two harness checks.

The checkpoints (8.5 GB each) stay on the cluster under
`/home/gotham/tmp/plugrl/e32/<arm>/ck/fpo/pi0-policy/<arm>/`.
