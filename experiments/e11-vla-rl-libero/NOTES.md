# E11 notes

Dated records of what happened while running this experiment. `PROTOCOL.md` is not edited. Anything that bears on how its results should be read is written here instead, with its date.

---

## 2026-09-13 - The first inference against a real pi0.5 crashed the server

The first smoke cell ran `Pi0Policy` through the real server loop for the first time: `libero_spatial`, one episode per task. It was not one of the protocol's cells. The server died on its first `infer`:

```
TypeError: Unsupported model observation type: <class 'list'>
```

openpi's `LiberoInputs` sets each `image_mask` entry to `np.True_` or `np.False_`, which is a numpy scalar, not an ndarray. PlugRL's `batch_aggregate` stacked only ndarrays, so each mask became a Python list, and the tensor conversion rejected it. `text` came through intact at every stage.

It is fixed in plugrl-server #14, commit `8b6aa27`, with a regression test. **Every E11 number comes from a server that includes that fix.** Each cell writes a per-file sha256 manifest of the source it ran to `results/environment-source-<cell>.txt`, with carriage returns stripped so each hash can be compared with the git blob.

**How the stopping rule was read.** `PROTOCOL.md` stops the experiment "if Stage A cannot run end to end". The crash happened in a harness check run before any protocol cell existed, so it is recorded here rather than treated as a Stage A result.

## 2026-09-13 - Workers from a killed run joined the next server

The second smoke cell reported 10 episodes from its clients, while the server recorded 20.

**The extra ten came from the first smoke cell.** After that cell's server crashed, its env clients were killed by matching the CLI's name. LIBERO's multiprocessing workers run as `python -c from multiprocessing.spawn ...`, so they survived. An env client retries while its server is away, and these kept retrying for about half an hour. They joined the next server the moment it listened, ran one episode per task, and exited. Their ten `env_steps` match the ten extra episode lengths exactly.

**Only the server's own record could show this.** The client-side rows of the cell were clean. The harness's reconciliation said "false", but for a different reason: it compared the server's step counter, which the eval algorithm leaves at 0, so it could not have been true for any eval cell.

**A second leak.** The same cell also left its server running after teardown, because the harness killed the subshell rather than the server. The next launch found it still holding GPU 0.

**Changes made before any protocol cell ran:**

- **The harness now refuses to start** while any LIBERO client or pi0 server process is alive. It runs the server and the clients each in their own session, tears down whole process groups, stops the clients if the server dies, and checks that nothing survived.
- **Reconciliation compares the right quantities.** It compares the server's episode count with the clients', and the sum of the server's episode lengths with the clients' environment steps. Replayed on the contaminated output, it reports 20 against 10 and 2244 against 1139, and marks every row invalid.
- **`valid` now includes the accounting check.** `PROTOCOL.md` says a row whose step accounting disagrees is invalid. The first version of the harness did not implement that; this one does.

**A third smoke cell ran on 2026-09-14 under the hardened harness.** It completed 10/10 episodes with client exit 0. The server recorded 10 episodes and 1052 steps against the clients' 10 and 1052. There were zero missing-step-state warnings and zero reconnects, and teardown was clean. Smoke rows are written outside `results/` and are not part of any statistic.

## 2026-09-14 - LIBERO renders on a different GPU than the harness names

The harness gives the env clients `CUDA_VISIBLE_DEVICES=1` and `MUJOCO_EGL_DEVICE_ID=1`. During the third smoke cell, `nvidia-smi` showed the ten clients' EGL contexts on GPU 2, at 526 MiB each, not on GPU 1. EGL enumerates devices in a different order from CUDA.

All eight GPUs on the node are the same model, so this does not change anything measured. It matters only for knowing what else shares that GPU.

## 2026-09-14 - Stage A finished; Stage B chose task 8

Both Stage A cells are valid:

- **A1**, `libero_spatial`: 99 of 100.
- **A2**, `libero_10`: 185 of 200.

In both, the server's record reconciles exactly with the clients', and there are no missing-step-state warnings. The numbers and predictions belong to `FINDINGS.md`.

**Stage B.** By its rule, Stage B chose `libero_10` **task 8**, "put both moka pots on the stove", at 11 of 20. It was the only task between 0.10 and 0.80.

## 2026-09-14 - Three defects stood between FPO and pi0, before any Stage C run

`FPOAlgorithm` had only ever trained `FPOPolicy`, an MLP. Stage C's harness checks ran FPO on the real `pi05_libero` policy with a 64-transition buffer. They are not protocol runs, and nothing from them is evaluated. They found three defects, each fixed in plugrl-server with a test before the next check.

**1. FPO could not store a tree observation.** Fixed in #16.

`derive_train_state` cast the observation to float32, which raises on a dict. `_refresh_epoch_value_targets` sliced it with `obs_all[i:j]`, which a dict does not support. The first of these runs in `FPOAlgorithm`'s constructor, so an FPO server for pi0 could not start. No FPO test existed.

**2. Values in bfloat16 could not become numpy arrays.** Fixed in #17.

`Pi0Policy` builds its value head in bfloat16. The first learn stopped at `value_batch.detach().cpu().numpy()` with `Got unsupported ScalarType BFloat16`. `GAEBuffer` does the same conversion right after.

**3. A learn step left most of the action expert where it was.** Fixed in #18.

After a check of 32 optimizer steps at a learning rate of 1e-5, the checkpoint was compared with the base weights:

| action expert, by dtype | params | changed |
|---|---|---|
| bfloat16 linear layers | 311,427,072 | 4.7% |
| float32 norms | 116,505,600 | 99.1% |
| bfloat16 `lm_head` | 263,323,648 | 0.0% |

- **Why the linear layers barely moved.** The spacing between neighbouring bfloat16 values near the median weight, 0.026, is about 1.2e-4. An Adam step at 1e-5 rounds away, and nothing kept it in higher precision.
- **Why `lm_head` did not move at all.** It is not on the denoising path, so it gets no gradient.

The consequence: without the fix, Stage C would have trained the expert's float32 norms and projections, not the expert this protocol describes, and nothing would have said so.

**The fix.** FPO's optimizer now steps float32 copies of every trainable half-precision parameter, and rounds them back into the model after each step. `Pi0Policy` also freezes the expert's `lm_head`. The forward is unchanged, and Stage C starts from exactly the weights Stage A evaluated.

A harness check with the fix repeated the 32 steps at a learning rate of 1e-5:

| action expert, by dtype | changed before | changed after |
|---|---|---|
| bfloat16 linear layers | 4.7% | 41.6% |
| float32 norms | 99.1% | 99.1% |
| `lm_head` | 0.0% | 0.0% (now frozen) |

The linear layers that still have not changed hold their progress in the float32 copies. Their next change arrives once the accumulated steps cross a bfloat16 value. Stage C takes 2048 steps per learn iteration, not 32.

GPU 0 peaked at 20,440 MiB, against 19,208 MiB without the fix.

**Status of these fixes.** They are code defects found and fixed before any Stage C run. None is one of Stage C's two allowed adjustments. **Every Stage C run uses a server that includes all three**, checked file by file against the plugrl-server commit it names.

## 2026-09-14 - Rules for Stage C that the protocol does not state

`PROTOCOL.md` fixes Stage C's hyperparameters, its adjustment limit, and "50 episodes for the fine-tuned policy and 50 for the baseline". It does not say:

- how long to train;
- which checkpoint to evaluate;
- how the training environment is run;
- which initial states the evaluation uses.

These rules were proposed and agreed on 2026-09-14, before any Stage C run.

**Training length and the evaluated checkpoint.**

- **Length.** Train for exactly **10 learn iterations** of 4096 transitions, that is `global_steps = 40960`.
- **What gets evaluated.** Only the checkpoint written when the run stops is evaluated. Intermediate checkpoints, saved every 5 iterations, exist for crash recovery and are never evaluated.
- **If the budget runs out.** The last saved checkpoint is evaluated and the result is labelled incomplete.

**Training environment.** Ten env client processes all run task 8, with initial states randomised by the runner's per-process seeds. LIBERO has 50 initial states per task, so training sees the states the evaluation uses.

**Evaluation.** One env client process runs 50 episodes with initial states in order, 0 to 49. This is openpi's LIBERO procedure. The baseline and the fine-tuned policy face exactly the same 50 initial states.

**Hyperparameters.** The protocol's starting values, with `batch_size` 8 per `AMENDMENT.md`.
