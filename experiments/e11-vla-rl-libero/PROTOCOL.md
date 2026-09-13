# E11 measurement protocol (pre-registered)

**Written 2026-09-13, before any success rate has been measured.** Smoke tests
of the environment and of weight loading were run before this was written: a
LIBERO env was created, reset and stepped once, and the checkpoint was loaded
strictly into the model. Neither measures a success rate, and neither touches
the question below.

**This file must not be edited after the first data point exists.** If a rule
turns out to be flawed, write a new version saying why the old one was
unusable, and keep both.

---

## The question

The experiments index has said since it was written that no VLA has been
trained through this system. E10 executed a VLA's forward pass. It did not run
one through the loop PlugRL exists for - an environment client talking to a
training server over the protocol - and it trained nothing.

E11 asks two things, in order:

1. **Does a VLA run end to end through PlugRL, and succeed at LIBERO tasks at
   roughly the rate it is published to?** If not, the pipeline is broken and
   nothing downstream means anything.
2. **Does FPO fine-tuning through PlugRL raise the success rate on a task
   where the baseline leaves room to?**

### A claim this relates to but cannot settle

`src/plugrl_server/policy/openpi/README.md` reports suite-level success rates
for a pi0.5-tiny policy before and after PPO, for example 0.51 to 0.752 on
LIBERO-10. No logs, configuration or checkpoint behind those numbers are in
this repository, so they cannot be reproduced from what is committed. E11 uses
a different model (full-size pi0.5) and a different algorithm (FPO), so it can
neither confirm nor refute them, and will not be presented as doing either.

## Held fixed

* **Policy:** `pi0-policy` with `--policy.name pi05_libero`, full size
  (`gemma_2b` backbone, `gemma_300m` action expert),
  `train_expert_only=True`, `denoising_steps=5`.
* **Weights:** the HuggingFace repository `sunshk/pi05_libero_pytorch`, a
  PyTorch conversion of openpi's `pi05_libero` in openpi's own checkpoint
  layout. **This is not an official release.** Verified before this protocol:
  812 tensors load strictly into `PI0Pytorch`, and the one tensor the model
  expects that the file lacks, `embed_tokens`, shares storage with `lm_head`,
  which is present. Its `norm_stats.json` is byte-identical to openpi's copy
  on Google Cloud Storage (sha256 `b3a44bb2...`).
* **Tokenizer:** openpi fetches `gs://big_vision/paligemma_tokenizer.model` at
  policy construction, and the cluster cannot reach Google Cloud Storage. The
  same object, 4,264,023 bytes with sha256 `8986bb4f...`, is placed in
  openpi's download cache ahead of time, which openpi's own cache lookup then
  returns without a network call. No code is changed.
* **Environment:** PlugRL's `Libero-v1`, on the LIBERO fork PlugRL pins
  (`CTP314/LIBERO@f3bf9428`), with assets from `lerobot/libero-assets`.
  Verified before this protocol: all 585 asset paths match the fork's, and
  every one of the 326 size differences is exactly the fork checkout's CRLF
  count.
* **Known environment deviations**, stated now rather than discovered in the
  write-up. `mujoco` is pinned to 2.3.7, because 3.13 fails robosuite 1.4.1's
  joint-type assertion. `jax` is 0.5.3 on CPU, matching openpi's lockfile.
  `lerobot` is installed from source at the commit PlugRL pins, because no
  PyPI release has the module layout openpi imports. The `datasets` family is
  taken from openpi's lockfile, because newer `fsspec` excludes every
  `datasets` release that has the API `lerobot` imports. The full resolved
  environment of both processes is recorded at run time in
  `results/environment-server.txt` and `results/environment-client.txt`.
* **Hardware:** RTX 3090, 24 GB.

## Stage A - baseline, no learning

The `eval` algorithm. One env client process per task, via
`--runner.pass-proc-id`, so each process runs a different task of the suite.
Per-suite episode step caps are taken from openpi's LIBERO example and
recorded with the results.

| cell | suite | tasks | episodes per task |
|---|---|---|---|
| A1, the control | `libero_spatial` | 10 | 10 |
| A2 | `libero_10` | 10 | 20 |

Success rate per task, with a Wilson 95% interval.

## Stage B - choosing the task, by a rule fixed now

From A2, choose **the task with the lowest success rate among those between
0.10 and 0.80 inclusive**. Below 0.10 there are too few successes to learn
from. Above 0.80 there is too little room to show a change.

* If no task falls in that range, choose the lowest non-zero task.
* If every task is at or above 0.90, **Stage C does not run**, and "no
  headroom on this suite" is the finding.
* Ties go to the lower task id.

The task is chosen by the data. The rule is not.

## Stage C - FPO fine-tuning on the chosen task

Starting values, **chosen without tuning, because no FPO run on a VLA exists
to tune from**:

| parameter | value | why not the default |
|---|---|---|
| `learning_rate` | 1e-5 | the default 3e-4 was set for a 272k-parameter MLP |
| `batch_size` | 32 | the default 1024 is sized for the same MLP |
| `num_updates_per_batch` | 4 | |
| `n_samples_per_action` | 4 | halves the default's memory |
| `buffer_size` | 4096 | |
| `clipping_epsilon` | 0.05 | default |

**At most two adjustments are allowed**, each recorded in `AMENDMENT.md`, with
the observation that forced it, before the run that uses it. Running out of
memory, or a loss that diverges within the first learn step, count as such
observations. A result that needed a third adjustment is reported as not
obtained.

Evaluation afterwards uses Stage A's procedure on the chosen task: **50
episodes for the fine-tuned policy and 50 for the baseline**, rerun at the same
N so the two are compared like for like rather than against A2's 20.

## Predictions, recorded before the data

* **P1, the control.** A1's mean success rate is within 10 points of openpi's
  published 98.8 for `libero_spatial`, that is, at least 88.8. *Falsified if*
  below. A failure here means the pipeline is wrong - observation transforms,
  image orientation, action post-processing - not the model.
* **P2.** A2's mean success rate is below A1's, the published ordering of the
  suites. *Falsified if* not. Overlapping intervals mean "cannot separate".
* **P3.** Fine-tuning raises success on the chosen task: the fine-tuned
  interval lies above the baseline interval. *Falsified if* the two 95%
  intervals overlap, which is the rule every experiment in this directory
  uses for ranges.
* **P4.** Correctness holds with a real VLA behind the server: zero
  missing-step-state warnings and exact step accounting in every run, as E9
  found for the MLP. *Falsified by a single occurrence.*

## Known limitations, stated in advance

* **An unofficial checkpoint.** A conversion error would show up as a failed
  P1, which is why P1 exists.
* **One task, one seed** for fine-tuning. The budget allows no more, and the
  result will say so rather than generalise.
* **The VLM backbone is frozen.** Only the action expert trains.
* **FPO is not PPO**, so nothing here is comparable to the README's +PPO
  column.
* **Untuned hyperparameters.** A negative P3 is a statement about these values
  on this task, not about FPO on VLAs.

## Stopping rules

* If the policy cannot be constructed, or Stage A cannot run end to end, that
  is recorded as the finding with the error, and the experiment stops.
* **If P1 fails, Stage C does not run.** Fine-tuning through a broken pipeline
  produces numbers, not results. The defect is found and fixed, and a new
  version of this protocol is written before anything continues.
* Budget: **8 GPU-hours for Stage A, 40 for Stage C.** A stage that exhausts its
  budget is recorded as incomplete, with what it did finish.

## Record format

```
results/stageA.tsv        cell suite task_id episodes successes sr ci_lo ci_hi valid
results/stageC_train.tsv  step episodes rolling_sr mean_reward learn_s gpu_mem_gb valid
results/stageC_eval.tsv   policy task_id episodes successes sr ci_lo ci_hi valid
```

`valid` is false for any run whose env clients did not all exit zero, or whose
step accounting disagrees. Invalid rows are kept and excluded from every
statistic.
