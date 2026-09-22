# E14 measurement protocol (pre-registered)

**Written 2026-09-20, before the run it describes.** A feasibility run and a
memory probe came first and are declared below rather than predicted.

This file must not be edited after the first registered data point.

---

## The question

E11 asked whether a real VLA can be trained through PlugRL's boundary and
whether it helps. It answered the first half and stopped at **one iteration of
ten**: the float32 master weights and Adam's two moments do not exist until the
first learn step, and the second step could not fit beside them on a 24 GB
card. It missed by 304 MiB.

So E11's negative result - 26 of 50 down to 0 of 50 - rests on **a single
gradient iteration**. One iteration destroying a policy is not surprising and
says little. The question this run exists to answer is the one E11 could not
reach:

**Over ten iterations, what does FPO do to a full-size pi0.5?** And, since E11
also observed that a collapse was reported rather than located: **when does it
happen, and does anything recover?**

### What this cannot settle

One task, one seed for the environment, one set of hyperparameters nobody
tuned, and a batch size of 8 forced by memory. A negative result here remains
a statement about these values on this task, exactly as E11's protocol said of
its own. Ten iterations is not convergence.

---

## Declared in advance: what was already known

Taken before this protocol was written, and not part of the registered result:

1. **A memory probe** on `pi05_libero`: the master copies and Adam's moments
   are 3,578 MiB, and placing them on a second card takes the model's card
   from 11,624 MiB to 8,046 MiB after one optimizer step. 905 MiB stays
   behind, being Adam's state for float32 parameters stepped in place.
2. **A two-iteration feasibility run** with the copies on `cuda:1`:
   both iterations completed, checkpoints written at steps 4100 and 8193, **no
   `OutOfMemoryError`**, server alive throughout, 9,044 s for the pair. Peak
   memory 24,095 MiB on the model's card and 17,781 MiB on the second - the
   latter transient, since the steady state there was about 6,300 MiB.
   Per-iteration wall clock was about 4,522 s against E11's 4,052 s, roughly
   **12% slower**, from wall clock alone and not a controlled comparison.

The feasibility run is why ten iterations is worth attempting. It is not
evidence about what ten iterations do to the policy, which is what is asked
here.

---

## Held fixed

Everything E11's Stage C fixed, so the two are comparable:

- **Policy**: `pi0-policy`, `pi05_libero`, full size, `train_expert_only`, 5
  denoising steps, from the same checkpoint directory E11 used.
- **Task**: `libero_10` task 8, the task E11's Stage B rule chose. Randomised
  initial states during training, as in E11.
- **Algorithm**: `fpo`, learning rate 1e-5, `batch_size` 8, 4 updates per
  batch, `n_samples_per_action` 4, `buffer_size` 4096, clipping epsilon 0.05.
  The batch size is E11's amended value, not the protocol's original 32.
- **Clients**: 10 processes, one environment each, `replan_steps` 5, seed 7.

Changed from E11, deliberately:

- **`master_weights_device=cuda:1`**, which is the point.
- **`--seed 7` on the server.** E12 found the flag reached only the run's name;
  it now seeds Python, NumPy and torch. With ten clients this does **not** make
  training reproducible, because batch composition depends on arrival timing.
  It is set for the record and because the evaluations below need it.
- **`save_interval` 1**, so every iteration leaves a checkpoint. E11 used 5 and
  its first attempt crashed with nothing to evaluate.

## The run

One training run of **10 iterations**, `global_steps = 40960`.

Then evaluations, each one env client process over **50 episodes** of task 8
with initial states 0 to 49 in order - E11's procedure, and openpi's:

| cell | policy |
|---|---|
| `eval-baseline` | the unmodified checkpoint |
| `eval-iter01` | after 1 iteration |
| `eval-iter02` | after 2 |
| `eval-iter05` | after 5 |
| `eval-iter10` | after 10 |
| `eval-iter10-repeat` | after 10, again, same seed |

Four training checkpoints rather than ten: enough to say whether a collapse is
immediate or gradual and whether anything recovers, without spending five and a
half hours to evaluate points between which nothing is expected to differ.

**`eval-iter10-repeat` is not about the policy.** It checks the seeding fix
under real conditions: same checkpoint, same seed, same states. Nothing has
verified that fix outside a unit test.

## Predictions, recorded before the data

1. **All ten iterations complete.** Falsified by any `OutOfMemoryError`, or by
   the server exiting before step 40960. Two iterations are known to fit; ten
   allocate no more state, but the allocator may still grow.
2. **Iteration 10 does not beat the baseline.** Falsified if its Wilson 95%
   interval lies entirely above the baseline's.
3. **The collapse is already complete at iteration 1** - iteration 1 scores at
   or below 2 of 50. Falsified at 3 or more. E11 measured 0 of 50 there, on a
   different run of an unseeded policy.
4. **Nothing recovers**: no evaluated iteration scores above the baseline's
   lower interval bound, 0.3851. Falsified by any that does.
5. **`eval-iter10-repeat` returns exactly the same successes as `eval-iter10`.**
   Falsified by any difference. This is the seeding fix or it is not.
6. **Per-iteration wall clock stays under 1.2x E11's 4,052 s**, that is under
   4,862 s. Falsified above it.

Predictions 2, 3 and 4 are expectations of failure. If any is falsified the
result is more interesting, not less.

## Known limitations, stated in advance

- **One seed for the environment, one task.** Nothing here supports a claim
  about FPO on VLAs in general.
- **Training is not reproducible** even with the seed set, because ten clients
  batch by arrival. Only the single-client evaluations are.
- **The baseline is re-measured** rather than taken from E11, because E11's
  evaluations predate the seeding fix and were not reproducible. Its 26 of 50
  is the number to compare against, not to assume.
- **Ten iterations at batch size 8** is 81,920 sampled actions of gradient
  signal. That is small, and the protocol says so before seeing the result.
- **The second card is part of the configuration now.** A reader with one GPU
  cannot reproduce this run; they can reproduce E11's single iteration.

## Stopping rules

- The training run runs once. If it dies, the cause is recorded and
  investigated in the findings; it is not restarted with different values,
  which would be a new protocol. One restart is allowed if the cause is
  external - another job taking the card, the machine rebooting - and that
  restart is recorded with its reason.
- Each evaluation runs once, except `eval-iter10-repeat`, which exists to be a
  second run.
- If the cluster's GPUs are occupied by other work, the run waits rather than
  squeezing beside it.

## Record format

- `results/train.tsv`: iterations completed, wall clock, peak memory per card
- `results/eval.tsv`: one row per evaluation - policy, episodes, successes,
  success rate, Wilson 95% interval
- `results/*.log`: the harness output of each cell as produced
- `FINDINGS.md`: what was asked, what came back, and what it does not support
