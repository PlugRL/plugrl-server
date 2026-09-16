# E11: a VLA trained through PlugRL

**A full-size pi0.5 ran end to end through PlugRL's process boundary - inference, feedback and a training step - and the server's record of what happened reconciles exactly with the clients'. The reinforcement learning result is negative: one FPO iteration took the chosen task from 26 of 50 to 0 of 50, and the run is incomplete, because a second learn step does not fit in 24 GB.**

Pre-registered in [`PROTOCOL.md`](PROTOCOL.md) on 2026-09-13, before any success rate had been measured. [`NOTES.md`](NOTES.md) carries the dated record of everything that happened on the way, and [`AMENDMENT.md`](AMENDMENT.md) the one adjustment the protocol allows that was used.

## What ran

| | |
|---|---|
| Policy | `pi0-policy`, `pi05_libero`, full size, `train_expert_only`, 5 denoising steps |
| Weights | HuggingFace `sunshk/pi05_libero_pytorch`, an unofficial PyTorch conversion. 812 tensors load strictly, and its `norm_stats.json` is byte-identical to openpi's |
| Environment | PlugRL's `Libero-v1` on `CTP314/LIBERO@f3bf9428`, assets from `lerobot/libero-assets` |
| Hardware | RTX 3090, 24 GB, for the server; LIBERO renders on a second card |
| Stage A code | plugrl-server `8b6aa27`, matched file for file against each cell's recorded manifest |
| Stage C code | plugrl-server `0c36c0e` in 63 of its 65 files. `cli.py` and `server/websocket_agent_server.py` were the pre-#20 versions: #20 was merged but never deployed, so the connection behaviour described below is the old one |

Every cell writes a per-file sha256 manifest of the source it ran, in `results/environment-source-<cell>.txt`. The claims above are that comparison, not an assumption.

## Stage A: the control

| cell | suite | episodes | successes | success rate | 95% interval | wall |
|---|---|---|---|---|---|---|
| A1 | `libero_spatial` | 100 | 99 | **0.990** | [0.9455, 0.9982] | 559 s |
| A2 | `libero_10` | 200 | 185 | **0.925** | [0.8800, 0.9540] | 2,022 s |

openpi publishes 98.8 and 92.4 for these suites at 30k steps. Both cells land there.

In both, the server's own record reconciles exactly with the clients': A1, 100 episodes and 10,632 steps on each side; A2, 200 episodes and 53,945 steps. Zero missing-step-state warnings, zero reconnects, every client exited zero.

## Stage B: the task the rule chose

`libero_10` task 8, "put both moka pots on the stove", at 11 of 20. It was the only task inside the rule's window of 0.10 to 0.80; every other task scored 0.85 or above. The rule was fixed before A2 ran.

## Stage C

### The two attempts

| | attempt 1 (`C_train`) | attempt 2 (`C_train2`) |
|---|---|---|
| started | 2026-09-16 18:18 | 2026-09-16 20:27 |
| iterations completed | 1 of 10 | 1 of 10 |
| collection, iteration 1 | 562 s | 650 s |
| learn, iteration 1 | 3,490 s | 3,489 s |
| success during collection | 0.632 | 0.513 |
| checkpoint | none - the interval was every 5 iterations | step 4096, saved 21:42:20 |
| died | 19:37:03, collecting iteration 2 | 21:51:46, starting iteration 2's learn |
| GPU 0 at death | 24,098 MiB of 24,125 | 24,018 MiB of 24,125 |

Both deaths are the same allocation: 124 MiB for an attention matmul in the action expert.

**What the memory did.** Before the first learn, collection ran at about 8,700 MiB. The first learn peaked at 23,558 MiB in attempt 1.

- **Attempt 1** then stayed there: PyTorch keeps a learn step's blocks in its caching allocator, so collection resumed against a card with nothing left and died four minutes later. 21.48 GiB was allocated and 1.33 GiB reserved but unused.
- **Attempt 2** released the cache after each learn (plugrl-server #21). Standing memory fell to **17,104 MiB**, collection for iteration 2 completed - the ten clients reported about 815 inference calls each, filling the second buffer - and the run died seven seconds later as the second learn began. This time 23.05 GiB was allocated and only 73 MiB reserved but unused: the memory was in use, not stranded.

The difference between 8,700 MiB and 17,104 MiB is what a first learn step leaves behind and cannot give back: float32 master copies of the trainable weights, and Adam's two moments, which do not exist until the first step. **The first learn fits in 24 GB precisely because that state has not been allocated yet; the second one has to fit alongside it, and does not.**

### The evaluations

Both ran one env client process over 50 episodes on task 8, with initial states 0 to 49 in order - the same 50 states for both policies, which is also openpi's LIBERO procedure.

| policy | episodes | successes | success rate | 95% interval |
|---|---|---|---|---|
| baseline | 50 | 26 | **0.520** | [0.3851, 0.6520] |
| fine-tuned, **incomplete** - one iteration of ten | 50 | 0 | **0.000** | [0.0000, 0.0713] |

Both are valid: 50 episodes and 23,735 steps on each side for the baseline, 50 and 26,000 for the fine-tuned policy, zero missing-step-state warnings, zero reconnects, both clients exited zero. The evaluated checkpoint is step 4096, sha256 `b8c25dee...`.

The baseline was measured once, before the first attempt, and not repeated for the second: neither #20 nor #21 touches the evaluation path.

## Predictions

* **P1 confirmed.** A1's 0.990 is within 10 points of openpi's published 98.8; the threshold was 0.888. The pipeline - observation transforms, image orientation, action post-processing - is not wrong.
* **P2 cannot be separated.** A2's 0.925 is below A1's 0.990 as predicted, but the intervals overlap - A1's lower bound is 0.9455 against A2's upper bound of 0.9540 - and the protocol's rule for overlapping intervals is that this scores nothing either way.
* **P3 falsified.** The prediction was that the fine-tuned interval would lie above the baseline's. It lies below it, and the two do not touch: [0.0000, 0.0713] against [0.3851, 0.6520]. One FPO iteration at these settings did not improve the policy on this task; it destroyed it.
* **P4 holds where it can be tested, and is untestable in the training runs.** Zero missing-step-state warnings in all four cells. Step accounting is exact in both evaluations. In both training runs it is **unavailable**: the server died, so the harness killed the clients, and a killed client never writes the final summary the comparison needs. That is a gap in the measurement, not a disagreement.

## What P3's result does and does not mean

**What it shows.** A complete FPO iteration ran: 4,096 transitions collected across a process boundary, 2,048 optimizer steps at a learning rate of 1e-5, a checkpoint written, and a policy that measurably changed. The loop works; the change was for the worse.

**What it cannot distinguish.** The protocol evaluates only the final checkpoint, and this run has exactly one. Nothing was measured between 0.520 and 0.000, so the run cannot say whether the collapse came from the first few steps or accumulated over 2,048 of them. In attempt 1's learn, 38.5% of samples hit the clipping bound of 0.05, which says the policy moved a long way in one iteration, but that is a description, not a diagnosis.

**What it is not.** It is not a statement about FPO on VLAs. One task, one seed, untuned hyperparameters, an action expert in bfloat16, and a batch size of 8 forced by memory - the protocol said in advance that a negative P3 is a statement about these values on this task.

## What had to be fixed before a VLA could train here

Four defects, none of which any test or any smaller policy had exposed. Each was found by running this experiment, fixed with a regression test, and merged before the run that needed it.

| PR | defect | what it cost |
|---|---|---|
| #14 | openpi's image masks are numpy scalars, and PlugRL's batching turned them into Python lists | the server died on its first inference against a real pi0.5 |
| #16 | FPO stored an observation as one float array, which a tree of images, masks and token ids is not | an FPO server for pi0 could not start |
| #17 | a bfloat16 value head cannot become a numpy array | the first learn stopped at the first value refresh |
| #18 | an Adam step of 1e-5 rounds away in bfloat16, and nothing kept it | 95% of the action expert's linear weights did not move |

The fourth is the one that would have produced a plausible, wrong result. It does not crash and it is invisible in a loss curve. It was found by comparing a checkpoint with the base weights, parameter by parameter, over the same 32 optimizer steps:

| action expert, by dtype | parameters | changed before #18 | after |
|---|---|---|---|
| bfloat16 linear layers | 311,427,072 | 4.7% | **41.6%** |
| float32 norms | 116,505,600 | 99.1% | 99.1% |
| bfloat16 `lm_head` | 263,323,648 | 0.0% | frozen: it is not on the denoising path |

Two more were found by the Stage C runs themselves: **#20**, so that a learn step is not mistaken for a dead client, merged after attempt 1; and **#21**, releasing the allocator's cache after a learn, open at the time of writing. #21 was deployed to the cluster for attempt 2 and is why its collection reached the second learn at all. #20 was not deployed, so both training runs still show the old behaviour.

## Where an hour of this training goes

One iteration of 4,096 transitions, measured by the server:

| phase | seconds | share |
|---|---|---|
| collection | 562 | 13.9% |
| learn | 3,490 | 86.1% |
| — prefix forward (`obs_cache`) | 1,389 | 44.6% of the learn |
| — per-epoch value refresh | 1,396 | 44.9% |
| — backward and optimizer | 233 | 7.5% |

Two things follow. The env clients idle for 86% of the wall clock, because the model lock serialises learning and inference. And the learn spends about four fifths of its time recomputing the prefix of a **frozen** VLM - the same observation's prefix, once per minibatch, four times per iteration. Neither is addressed here.

## Three things these runs showed about the server itself

**A learn step looked like a dead client.** The connection handler waits 60 s for feedback after sending an action. A learn step here takes 3,490 s. Seconds into the first one the server closed all ten connections with "Timed out waiting for feedback", and the clients reconnected on their own - discarding, per SPEC.md section 7.6, the feedback each was holding. Four such timeouts per training run, 420 client reconnect attempts in attempt 2. This is not E8's question: E8 asked whether a long learn trips the WebSocket keepalive and measured that it does not. This deadline is the server's own. Fixed in #20, after these runs.

**`ss` is blind inside this container**, which makes the harness's port check useless: `ss -tln` lists nothing at all, not even sshd's own listener, so `free_port` has never actually found a port to free. The listener was confirmed instead through `/proc/net/tcp` and a TCP connect. The harness scripts are committed as they ran and were not edited to match this finding.

**The server's connection count inflates transiently.** During the first learn it read 16 while ten clients were live: handlers of closed connections had not yet unwound past the model lock, and reconnected clients were already counted. It self-corrects. It matters because the inference batch threshold reads that count - but no stall followed, zero "infer queue has been waiting" warnings, so nothing was changed on that evidence.

## Budget

The six protocol cells spent about 4.5 hours of client wall clock: A1 559 s, A2 2,022 s, the baseline evaluation 1,937 s, two training attempts of 4,716 s and 4,838 s, and the fine-tuned evaluation 1,962 s. With model loading and seven harness checks that are not protocol runs, the experiment used roughly 13 to 14 GPU-hours of the 40 the protocol allows.

**It did not stop for lack of budget.** It stopped because a decision taken on 2026-09-16, before the restart and recorded in `NOTES.md`, allowed one restart after a crash, and the second attempt died the same way.

## Known limitations, as stated in advance

* **An unofficial checkpoint.** A conversion error would have shown up as a failed P1, which is why P1 exists.
* **One task, one seed** for fine-tuning.
* **The VLM backbone is frozen.** Only the action expert trains.
* **FPO is not PPO**, so nothing here is comparable to the README's +PPO column.
* **Untuned hyperparameters.** A negative P3 is a statement about these values on this task.

## What would be needed to answer the question properly

Not attempted here, and none of it is a small change:

* **Memory.** A second learn step does not fit beside its own optimizer state on a 24 GB card. Gradient checkpointing, a smaller batch, or sharding the optimizer would each make room; all three change what is being measured.
* **Evaluation between iterations**, so that a collapse can be located rather than observed.
* **More than one seed and more than one task**, before any claim about FPO on VLAs.
