# E14: a pi0.5 policy collapses in one FPO iteration, and five explanations have been refuted

> **Corrected 2026-09-22 and again 2026-09-24, after this document was
> merged.** The version that merged was titled "one FPO iteration destroys the
> policy" and read as a result about FPO. It should not have. This project did
> not write FPO and did not train pi0.5; when a published algorithm applied to
> a working model destroys it, the first hypothesis is a defect in how we are
> applying it, and that hypothesis was missing from the list this document
> checked.
>
> The first correction named a specific defect and called it the leading
> explanation. **It was then tested and refuted, along with four others.** The
> second correction records that, and withdraws a claim about the
> evaluation's reproducibility that does not survive either. No measurement
> in this document has changed; what changed twice is what they are allowed
> to mean.

**The baseline scores 29 of 50. After one iteration of FPO as configured here
it scores 0 of 50, and it is still 0 of 50 after nine. The training run's own
metrics - not read until after this document first merged - show the collapse
happening live: `rollout/success` goes 0.65, 0.026, 0, with no saving or
reloading involved. Five candidate causes have since been tested one knob at
a time and every one is refuted: advantage normalisation scope, a quarter of
the gradient steps, a tenth of the learning rate, a trust region swept across
25x, and a critic warmup that leaves the value head a full iteration of
returns ahead. The cause is not known. What is characterised is the
phenomenon, and separately a defect in the evaluation harness that the
investigation turned up on the way.**

Pre-registered in [`PROTOCOL.md`](PROTOCOL.md) on 2026-09-21. Five amendments,
each dated and written before the data it bears on, are in
[`AMENDMENT.md`](AMENDMENT.md).

## What ran

| | |
|---|---|
| Server | `pi0-policy` / `pi05_libero` / `fpo`, buffer 4096, batch 8, 4 samples, `--seed 7` |
| Master copies | `master_weights_device=cuda:1`, the second card, which is what made a second learn step possible at all |
| Clients | 10 processes, one environment each, `libero_10` task 8, `replan_steps` 5, seed 7 |
| Server code | `main` at 32bc21a, which **deploys #20** - E11 did not have it (amendment 1) |
| Evaluations | one client process, 50 episodes, initial states 0-49 in order, seed 7 |
| Machine | cluster `qz103`, three cards chosen at launch from those actually free. Shared throughout with other users' jobs |

## The measurement

| policy | episodes | successes | rate | Wilson 95% |
|---|---|---|---|---|
| **baseline** | 50 | **29** | 0.580 | 0.442 - 0.706 |
| iter01 | 50 | **0** | 0.000 | 0.000 - 0.071 |
| iter02 | 50 | **0** | 0.000 | 0.000 - 0.071 |
| iter05 | 50 | **0** | 0.000 | 0.000 - 0.071 |
| iter09 | 50 | **0** | 0.000 | 0.000 - 0.071 |
| iter09-repeat | 50 | **0** | 0.000 | 0.000 - 0.071 |
| baseline-repeat | 50 | **29** | 0.580 | 0.442 - 0.706 |

Rows in [`results/eval.tsv`](results/eval.tsv). Training in
[`results/train.tsv`](results/train.tsv).

`iter09` stands in for the registered `iter10`; the tenth checkpoint does not
exist as a usable file, and the substitution is recorded in amendment 4 rather
than presented as what was asked for.

### The zeros are not a loading failure

Four identical numbers invite a dull explanation - that no checkpoint was
loaded, that the same one was loaded four times, or that saving and reloading
a policy damages it. Each was checked.

**A fifth explanation was not on that list and should have been the first
one: that our own use of the algorithm is defective.** Ruling out four
alternatives is worth nothing if the likely fifth is absent, and it was
absent here because the document was framing the result as a fact about FPO
rather than about this implementation of it. It is addressed in
[What the training metrics show](#what-the-training-metrics-show-and-the-defect-they-point-to),
and it is now the leading explanation. The checks below stand; they were just
not the whole list.

**Four different checkpoints were loaded.** The harness records a sha256 per
cell and they differ:

```
iter01  3d7593ebd9dc582f        iter05  795bbc92123186a9
iter02  48d3e67263d96794        iter09  19a551f527520544
```

The servers logged the paths they were given (`.../4100`, `.../36870`), the
baseline cell logged `policy_checkpoint_path=None`, and that cell - same
harness, same task, same 50 states - returned 29.

**The weights are finite and they moved.** Across the four evaluated
checkpoints, 0 of 821 tensors hold a NaN or an Inf, so this is not numerical
divergence. Between iteration 1 and iteration 9 the relative distance
`||b-a|| / ||a||` over all 821 shared tensors is **0.0045**, and the tensor
that moved furthest is **`critic.mlp.0.bias`, by 55%**. Training is changing
the policy, and the value head - which starts from a random initialisation -
is still moving long after the policy has collapsed.

**The round trip is symmetric and strict.** FPO's `create_checkpoint` saves
`self.policy.state_dict()`, the live policy that generated the rollouts, not
the float32 master copies. `Evaluation.load_checkpoint` calls
`self.policy.load_state_dict(...)` on the same kind of object with PyTorch's
default `strict=True`, so a missing or unexpected key raises rather than
passing silently - and the checkpoints include `critic.*` keys, which the
evaluated policy therefore also has, or the load would have failed. Every
evaluation exited 0.

A strict load either reconstructs the stored tensors or raises; the place such
a round trip can still lose something is **tolerance** in the path - a
`strict=False`, a key remapping, or a dtype cast, since `load_state_dict`
copies into the existing parameter and casts to its dtype. There is no
remapping and no `strict=False`, and the cast is ruled out by measurement: the
stored dtypes are 122 float32 and 699 bfloat16 tensors, which is the policy's
own mixture, because what was saved is that policy's `state_dict()`. Storage
and destination match element for element.

Even so, this is reasoning about the code path. The measurement that would
close it outright - saving an untrained policy through the same path,
reloading it, and checking that it still scores 29 - was not run, on the
judgement that an hour of contended GPU was not worth confirming what a strict
load with matching dtypes already forces. That is a judgement, and it is
recorded here rather than left as an implied control.

## The training run

Nine complete iterations, and a tenth that computed and lost its checkpoint.

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| wall s | 4696 | 4629 | 4680 | 4603 | 4639 | 4642 | 4636 | 4645 | 4660 |
| card 1 MiB | 18504 | 19946 | 20919 | 20086 | 19966 | 20066 | 20046 | 19986 | 19986 |
| card 2 MiB | 9071 | 9071 | 9925 | 9071 | 9071 | 9071 | 9071 | 9071 | 9071 |

**No `OutOfMemoryError` occurred.** Memory grew for three iterations, touched
20,919 MiB, and then came back to about 20,000 and stayed within 120 MiB of it
for the remaining six. The high-water mark is one iteration's transient, not
the level the run sits at.

The run ended at 43,206 s with exit status 124 - `timeout`. The env clients run
under `timeout 43200` in the harness, a twelve-hour budget, and the run needed
about thirteen. The server was alive throughout and shut down cleanly. The
tenth learn step finished and the checkpoint write was interrupted, leaving a
file 173,707,658 bytes short of the length its own header declares.

**That budget was checkable before the run and was not checked**: the
two-iteration probe's 4,522 s per iteration put ten iterations plus startup at
about 45,600 s against a 43,200 s bound. It is recorded as our failure in
amendment 4, not as a property of the method.

## What the training metrics show, and the defect they point to

The run wrote 46 scalar series to tensorboard throughout. The console was
quiet because the harness passes `--no-show-metric-table`, and this document
was first written without reading any of them - every claim in it was
reconstructed from checkpoints on disk while the record of what happened
inside the training loop sat beside them. The protocol asked for training
dynamics. They existed.

**The collapse is visible live, with no checkpoint involved:**

| per iteration | 1 | 2 | 3 | 4 | ... | 9 |
|---|---|---|---|---|---|---|
| `rollout/success` | **0.65** | **0.026** | **0** | 0 | 0 | 0 |
| `rollout/length` | 461.6 | 519.5 | 520 | 520 | 520 | 520 |
| `losses/value_loss` | 0.880 | 0.033 | 0.0019 | 0.0011 | | 0.000093 |

Iteration 1's rollouts are collected by the unmodified policy, before any
update, and score 0.65 - the baseline's 0.58, on different episodes. Iteration
2's are collected after exactly one learn step, and score 0.026. From
iteration 3 on, every episode runs to the 520-step limit and earns nothing.
The value loss falling to 9.3e-05 is not the critic learning; it is the critic
running out of anything to predict.

**The defect.** Advantages were normalised inside `_compute_loss`, which
receives one minibatch:

```python
advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
```

That is the ordinary arrangement, and it is sound at the batch sizes FPO
assumes - its own default `batch_size` is **1024**. A pi0.5-sized policy does
not fit those. This run used **8**, which `PROTOCOL.md` records as "forced by
memory". At 8, two things go wrong at once: the standard deviation of eight
samples estimates nothing, so dividing by it injects noise; and subtracting
their mean forces roughly half of every eight samples positive and half
negative, at unit scale, whatever the rewards were.

The metrics carry the signature. `fpo/advantages_std` is **exactly 1.000 at
every one of the nine iterations** - including the seven where
`rollout/reward` is 0 for every episode in the buffer. It could not have been
anything else: it was measured after the division. That is also why nine
iterations of metrics could not show that the buffer had stopped carrying any
signal at all.

`fpo/clipped_ratio_mean` sits at 0.34 to 0.44, so a third to nearly half of
every minibatch hits the clipping boundary at `clipping_epsilon` 0.05, and
`fpo/policy_ratio_mean` sits at 0.95 to 0.98, pressed against the lower bound.
`fpo/initial_cfm_loss_mean` rises 65-fold after the first learn step, from
0.00049 to 0.032, and keeps climbing to 0.135: the policy is drifting away
from the flow-matching solution it was pretrained to.

This run also sits far from FPO's design point in every direction that memory
forces: `buffer_size` 4,096 against a default of 983,040, `batch_size` 8
against 1,024, and 0.5 gradient updates per collected sample against 0.016.
Any of those could matter. The advantage normalisation is the one that can be
pointed at in a specific line and matched to a specific metric.

**It is not the cause. Tested, and refuted.** The change that normalises over
the buffer instead of the minibatch is in `fix/advantage-normalisation-scope`,
and a two-iteration run under this experiment's exact configuration returned
**0 of 50 and 0 of 50** - the same collapse. The change stands on its own
terms, because a standard deviation of eight samples estimates nothing, but it
is not what breaks the policy.

Four more hypotheses have been refuted the same way since, each a single knob
under E14's configuration, each evaluated over 50 episodes:

| what was changed | result |
|---|---|
| advantage normalised over the buffer, not the minibatch | 0, 0 |
| `num_updates_per_batch` 4 → 1, so 512 gradient steps instead of 2048 | 0 |
| `learning_rate` 1e-5 → 1e-6 | 0 |
| `clipping_epsilon` swept 0.05, 0.02, 0.01, 0.005, 0.002 | 0, 0, **8** (repeat 7), **2** (repeat 2), 0 |
| one iteration of critic warmup, actor frozen, then a real update | **31** for the frozen iteration, then **0** |

The clipping sweep is not monotone, so by the rule registered in
[E15's protocol](../e15-trust-region/PROTOCOL.md) before three of its five
points landed, it does not support a trust-region mechanism. The 8 and the 2
are reproducible rather than noise - their repeats returned 7 and 2 - and
remain unexplained. The critic warmup's 31 of 50 is the control working: with
the actor's gradients zeroed its weights cannot move, and the score returns to
the baseline's neighbourhood, which is what makes the 0 that follows readable.

So the collapse survives every knob that bounds how far a single iteration
moves the policy, and survives giving the value head a head start. **The cause
is not known.** What is characterised is the phenomenon: the frozen backbone
moves exactly 0, the trained parts move 0.66% to 2.4% whatever the knob, no
tensor holds a NaN, and `rollout/success` falls 0.65, 0.026, 0 inside the
training loop.

## The predictions

| | prediction | outcome |
|---|---|---|
| 1 | all ten iterations complete; falsified by any `OutOfMemoryError` or the server exiting before step 40,960 | **held, with a caveat**: no OOM, the server did not exit, step 40,960 was reached - but the tenth checkpoint did not survive the write |
| 2 | iteration 10 does not beat the baseline | **held** (on `iter09`): 0 of 50 against 29 |
| 3 | the collapse is already complete at iteration 1, at or below 2 of 50 | **held**: 0 of 50 |
| 4 | nothing recovers - no iteration above the baseline's lower bound of 0.3851 | **held**: every evaluated iteration is 0.000 |
| 5 | the repeat returns exactly the same successes | **held vacuously** - see below |
| 6 | per-iteration wall clock under 4,862 s | **held**: maximum 4,696 s, spread 93 s across nine |

Nothing was falsified. The protocol said of predictions 2, 3 and 4 that "if any
is falsified the result is more interesting, not less"; none was, so this is
the less interesting result, and the one that removes the doubt E11 left.

### Prediction 5 passed without testing anything

The registered repeat exists to test the seeding fix, not the policy. It
compares `iter09-repeat` against `iter09`, and both are 0 of 50 - but a policy
that fails every episode agrees with itself under any seed, a different seed,
or no seeding at all. The equality is guaranteed by the collapse and says
nothing about reproducibility.

This was noticed and written down in amendment 5 **before the cell ran**,
along with the remedy: re-run the one non-degenerate number the experiment
has, the baseline's 29 of 50, under the same seed and the same 50 states.

### The replacement test passed, and it is stronger than the one it replaced

**`eval-baseline-repeat` was not pre-registered.** It was added on 2026-09-22,
after the collapse had made the registered repeat vacuous, and both halves of
that sentence matter: a cell added once the data is in is normally the weaker
kind of evidence, and this one is not, for a reason that has nothing to do
with how it turned out.

`eval-baseline-repeat` returned **29 of 50**. Identical to the original
baseline, on the same 50 initial states under the same seed.

The two runs were not close together and not alike. The first ran on
2026-09-21 from 17:39, finishing in 4,101 s at a machine load near 193, beside
another user's 16 GB job. The second ran on 2026-09-22 from 19:55, finishing
in 3,694 s at a load near 290, beside six shards of a different job. Twenty-six
hours apart, different cards' neighbours, wall clock differing by 11%, and the
same 29.

That is a better test than the registered one would have been even if the
policy had not collapsed, because the two runs differ in everything except the
things that are supposed to determine the result.

### The generalisation drawn from it was wrong, and is withdrawn

This section originally continued: "every single-client evaluation this
project reports rests on a measurement that has now been shown to repeat."
**That does not follow and it is not true.**

The baseline cells run with no checkpoint. Every trained-policy cell loads
one. Those are different code paths, and only the first is reproducible:

| path | runs | results |
|---|---|---|
| no checkpoint | 3 | **29, 29, 29** |
| checkpoint, same file, bit-identical policy | 3 | **35, 28, 29** |

The three checkpoint runs loaded the same dumped base `state_dict`, so the
policy generating actions was identical in all three. An in-process
comparison shows that loading changes only the critic's four weight matrices,
leaves every actor tensor bit-identical, and touches neither the CPU RNG state
nor any module's training mode. The spread is therefore not the weights, and
it is not explained.

What survives is the narrower claim, now on three runs instead of two: **the
no-checkpoint evaluation repeats exactly.** What does not survive is carrying
that across to the path every trained-policy row in this document used.

The collapse is not threatened by this. A spread of seven episodes on a policy
scoring in the high twenties does not produce 0 of 50, and 0 of 50 was
measured eight separate times across five configurations. But every non-zero
number this document reports from a checkpoint carries an unquantified spread,
and that includes nothing in the table above except the baselines - which is
the one place it does not apply.

It also settles what the zeros mean. A baseline that repeats exactly makes
29 against 0 a difference between policies, not a spread between runs.

## What this does not support

- **Nothing about FPO.** This is the important one, and the merged version of
  this document failed it. FPO is published work that this project did not
  write, and pi0.5 is a model this project did not train. A collapse here is
  in the first instance a statement about *this implementation of FPO at this
  configuration*, and a defect in it has since been located. Nothing in this
  experiment supports a claim about the algorithm, and it would not even if no
  defect had been found: one task, one seed, no tuning, and no comparison
  against a known-good run of the same algorithm.
- **One task, one seed, one set of hyperparameters.** It is a statement about
  `pi05_libero` on `libero_10` task 8 at batch size 8 - where 8 is itself the
  thing that broke the advantage normalisation.
- **Nine iterations is not many.** 36,864 sampled actions of gradient signal
  is small, and the protocol said so before the data. A collapse this complete
  at iteration 1 is not obviously a question of scale, but this run cannot
  rule it out.
- **Not a tuned baseline.** No hyperparameter search was run. The collapse may
  be a learning rate away from something else entirely, and nothing here
  bounds how far.
- **Training is not reproducible** even with `--seed 7`, because ten clients
  batch by arrival time. Only the single-client evaluations are.
- **The load timeline for the run does not exist.** The cluster was shared
  throughout, the tunnel to it was down from 22:50 to the following afternoon,
  and the recorder that would have attributed wall clock to neighbours never
  covered the run. The 93 s spread across nine iterations is the evidence that
  contention did not matter here; there is no finer-grained account.
