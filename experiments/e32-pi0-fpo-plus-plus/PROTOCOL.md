# E32 measurement protocol (pre-registered)

**Written 2026-09-27, after the harness checks in `results/pilot.txt` and
before the registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

One FPO iteration takes pi0.5 on LIBERO-10 task 8 from about 29 of 50 to 0
(E14; E26's control, on today's code). It is not the size of the step (E15:
four ways of moving less all scored 0) but its direction (E21, E22), and
freezing the part of the expert E22 located it in only moves the damage
(E26).

FPO++ (Yi, Choi et al. 2026, *Flow Policy Gradients for Robot Control*;
reference implementation amazon-far/fpo-control) is FPO's authors' revision,
and its manipulation experiments fine-tune pretrained flow policies - this
situation. Its fine-tuning differs from the FPO here in how it scores an
action chunk and in how many ratios it forms. #74 adds both as options.

**Does FPO++'s way of scoring a chunk, or its per-sample ratio, stop the
first update from destroying pi0.5 - and which one?**

### Why the chunk loss is the suspect

FPO's ratio is `exp(old loss - new loss)` and its clip bounds that ratio: a
trust region measured in units of the CFM loss. On pi0.5 that loss is
unusually small, for three reasons that are properties of how FPO scores a
chunk, not of pi0.5:

* it averages over all 10 x 32 elements of the chunk, and only 35 are actions
  taken - the environment uses 7 dimensions and the client replans after 5
  steps;
* its default output mode supervises the velocity as an eps prediction,
  which weights the velocity error by (1 - t)^2 - zero at t = 1;
* pi0.5 fits its own samples closely.

E26's control logged the result (known item 2): an initial loss of 0.0005,
raised 276-fold by one learn step inside a clip of 0.05. FPO++ scores a
chunk by the velocity error's mean over the action dimensions, summed over
the chunk's steps, at t drawn uniformly on [0, 1).

### What this cannot settle

* Improvement. Each arm makes one policy update. An arm that holds has not
  been shown to learn anything; that needs more iterations, and is the next
  experiment if one holds.
* FPO++ in full. Its other settings are not taken (Design, below), and the
  reference itself never scores a partly executed chunk.
* One task, one seed, one evaluation per arm.

---

## Declared in advance: what was already known

1. **The null** (E15 correction 2): an unperturbed actor on the older code,
   seven evaluations, 28 to 37, 30.9 +/- 3.6. E26's `base`, on main as of
   2026-09-25 plus #60: 29.
2. **Vanilla FPO's first learn step on pi0.5**, read from E26's control
   tensorboard after it had run: `initial_cfm_loss_mean` 0.000515,
   `cfm_loss_mean` 0.142, `policy_ratio_mean` 0.878, `clipped_ratio_mean`
   0.601. Its evaluation: 0 of 50. The same statistics for FPO on Walker2d
   (E24, where it learns): 0.56 to 0.66, moving under 2%, ratio 0.989 to
   1.004, clipped 0 to 11%.
3. **One critic-only iteration first** changes nothing by itself: the
   comment on `n_critic_warmup_itrs` records a warmup iteration verified
   frozen at 31 of 50, followed by a real update at 0 of 50.
4. **The harness checks** (`results/pilot.txt`): both ran end to end on the
   real pi0.5. The second is `fpopp` at 64 steps per iteration: the
   critic-only iteration left every actor tensor bit-identical; under
   FPO++'s chunk loss the initial loss was 0.096 and 0.061 on its two learn
   steps - 120 to 190 times the vanilla 0.0005 - and its one policy update of
   32 gradient steps raised it 28%, clipped 16% of the per-sample ratios and
   moved the expert 0.22%. The registered run's update is 2,048 gradient
   steps.
5. **FPO++'s reference** (read before this file, amazon-far/fpo-control at
   8f460e4): per-sample ratio with the plain PPO clip (its README: ASPO
   "hurts performance" in fine-tuning); chunk loss = mean over dimensions,
   masked sum over steps; t uniform on [0, 1), one per sample; (t, eps) drawn
   once at collection; straight-through log-ratio clamp at 5; one
   value-only iteration first.

---

## Design

Four arms on qz103, each two FPO iterations of 4,096 steps. The first trains
only the value head (`--algo.n-critic-warmup-itrs 1`, as FPO++'s
`n_iterations_train_only_value = 1`); the second is the first policy update.
Everything else is E14's: `pi0-policy` `pi05_libero`, learning rate 1e-5, 4
updates per batch, clip 0.05, minibatch 8, 4 samples per action, float32
master weights on the second server card; ten LIBERO clients on task 8 with
randomised initial states, replanning every 5 steps, seed 7.

| arm | chunk loss | ratio |
| --- | --- | --- |
| `control` | FPO's: eps-supervised, t on the denoising grid, mean over all 320 elements | one per action |
| `chunk` | FPO++'s: velocity error, t uniform on [0, 1), mean over the first 7 dimensions, summed over the first 5 steps | one per action |
| `persample` | FPO's | one per sample, log-ratio clamped straight-through at 5 |
| `fpopp` | FPO++'s | one per sample |

FPO++'s chunk loss is `--algo.output-mode u --algo.no-discretize-t-for-training
--algo.cfm-loss-steps 5 --algo.cfm-loss-dims 7 --algo.cfm-loss-sum-over-steps`;
the per-sample ratio is `--algo.ratio-per-sample`. The 5 steps are the ones
the client executes, the 7 dimensions the ones LIBERO's action uses.

Kept at E14's values rather than FPO++'s, and not tuned: clip 0.05 (FPO++'s
manipulation runs use 0.01 to 0.03 - looser here, which works against the
FPO++ arms); 4 samples per action (8); 4 epochs of 512 minibatches of 8 (10
epochs, 8 minibatches of 375); advantages normalised over the whole buffer
(per minibatch - a minibatch of 8 is too small to normalise); squared error,
not Huber (a delta of 0.1 to 1, above nearly all of pi0.5's per-element
errors); no gradient clipping (1 to 25); no clamp on the old loss (4 per
step, far above pi0.5's).

Evaluations are E14's: task 8, fifty episodes, initial states 0 to 49 in
order, `runner.seed` 7; the untrained policy (`base`) and every arm's
iteration-2 checkpoint. `train.sh` and `eval.sh` are E14's harnesses derived
by `derive.py`, which changes only the code directory (through
`PYTHONPATH`), the output directory, an `ARM_FLAGS` pass-through and the
evaluation's source manifest, and refuses to write if any substitution
misses. Code: main 2bd2789 plus #74 (5272832), deployed LF as
`$R/plugrl-server-e32`.

---

## Checks

* **V1 - the first iteration trained only the critic.** In every arm's
  iteration-1 checkpoint, no tensor of the expert or its projections has
  moved from the base (`movement.py` against E15's base dump). An arm that
  fails is not read.
* **V2 - the flags took.** Every arm's config line in its server log shows
  its chunk loss, ratio, the warmup, learning rate 1e-5 and clip 0.05.
* **V3 - every evaluation valid**: fifty episodes, client exit 0, `valid`
  true in the harness's own table.

---

## Predictions, and what falsifies each

An arm **holds** at 20 of 50 or more (E26's threshold, three standard
deviations below the null's mean) and **collapses** at 5 or fewer. Between,
it is partial: reported, not read.

**P1 - the collapse reproduces with a critic-only iteration first.**
`control` collapses.

> Falsified by 6 or more; then P2 to P4 are not read.

**P2 - FPO++ prevents it.** `fpopp` holds.

**P3 - its chunk loss alone does.** `chunk` holds.

> Grounds: known items 2 and 4 - the vanilla loss is small enough that a
> clip of 0.05 bounds almost nothing; FPO++'s is not. Falsified by 5 or
> fewer.

**P4 - its per-sample ratio alone does not.** `persample` collapses.

> Grounds: a ratio per sample still compares FPO's vanilla loss, on the same
> scale. Falsified by 20 or more.

A holding arm that moved the expert less than 0.66% - the smallest
movement E15 found destructive - is reported as having held while moving
less than any destructive run so far: that alone cannot show the chunk loss
is the cause rather than the smaller step.

**Reported, not predicted:** `base`; every arm's relative movement per
module group after iteration 2; the policy update's `initial_cfm_loss_mean`,
`cfm_loss_mean`, `policy_ratio_mean` and `clipped_ratio_mean`.

---

## Declared deviations allowed in advance

1. One restart of any run or evaluation that dies for a reason outside the
   experiment, recorded in `AMENDMENT.md`.
2. Cards may be reassigned if the planned ones are occupied.

---

## Reading order

V1, V2, V3, then P1, P2, P3, P4, then the reported figures.
