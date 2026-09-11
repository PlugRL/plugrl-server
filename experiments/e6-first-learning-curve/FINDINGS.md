# E6: this project trains a policy, for the first time

2026-09-10 · Windows 11, CPU only (Intel Iris Xe, no NVIDIA GPU, torch
2.7.1+cpu) · plugrl-server @ cb5b369 · plugrl-env-client @ ff9d77e

## In one sentence

FPO on `HalfCheetah-v5`, three seeds, 500,000 environment steps each: episode
return goes from **-315 ± 26** to **1928 ± 224**, in about 100 minutes per
seed on a laptop with no GPU.

## Why this needed doing

Nothing in PlugRL had ever trained anything. Before this run:

* every tensorboard file on disk came from `DummyAlgorithm`, whose `learn` is
  a `sleep`; 45 of the 47 contained zero scalars, and the two that were not
  empty held one point each with `rollout/reward = 0.0`;
* **no model weights existed anywhere** in the git history of the four
  repositories;
* of six registered environment families, none could drive the one algorithm
  that has a real `learn_impl` - `classic-v1` is discrete-action, `d4rl-v1`
  does not install on Windows, and the rest need assets, a display or a
  Linux-only stack.

An RL framework that has never produced a learning curve is an assertion. So
this is the smallest run that turns it into a fact.

## Setup

`MuJoCo-v1` was added to `plugrl-env-client` for this, promoted from an
unregistered draft in `examples/`. `HalfCheetah-v5` has a 17-dimensional
observation and a 6-dimensional action, which are exactly
`FPOPolicyConfig`'s defaults (`obs_dim=17`, `action_dim=6`), so the shipped
flow policy points at it with no configuration.

| | |
|---|---|
| Algorithm | FPO, `--algo.buffer-size 4096`, otherwise defaults |
| Policy | `fpo-policy`, 272,423 parameters, `--policy.device cpu` |
| Environment | `HalfCheetah-v5`, one per client process, no rendering |
| Seeds | 0, 1, 2 - each seeding **both** the server (policy init) and the client (environment) |
| Steps | 500,000 per seed |

**`--algo.buffer-size` is the one non-default, and it is not optional.**
`FPOAlgorithm.should_learn()` is true when the rollout buffer is full or when
the run reaches its last step. At the default `buffer_size=983040`, a run of
under a million steps therefore learns **exactly once, at the very end** -
producing a single point rather than a curve. 4096 gives one update per 4096
environment steps, so 500,000 steps is 123 updates.

## Result

| seed | updates | first | final | best | mean of last 10 | wall clock |
|---|---|---|---|---|---|---|
| 0 | 123 | -303.9 | 2298.8 | 2331.4 | 2120.0 | 103 min |
| 1 | 123 | -296.4 | 2231.0 | 2231.0 | 1980.8 | 98 min |
| 2 | 123 | -344.2 | 1970.0 | 2126.3 | 1682.4 | 100 min |

Across seeds, using the mean of each run's last ten updates rather than its
final point, because the curve is noisy:

```
start   -314.8 +/- 25.7
end     1927.7 +/- 223.6
```

`summary.tsv` holds the per-update values for all three seeds; `summarise.py`
regenerates it and prints the curve.

```
step      seed0    seed1    seed2     mean
  4096   -303.9   -296.4   -344.2   -314.8
 40960   -102.4    -96.1   -194.5   -131.0   ###
102400    696.8    393.8     64.5    385.0   ############
204800   1268.9   1497.5    777.2   1181.2   ##########################
307200   1789.8   1049.1    850.2   1229.7   ###########################
409600   2210.5   2033.9   1481.6   1908.7   ######################################
500000   2298.8   2231.0   1970.0   2166.6   ###########################################
```

Seed 2 at step 102,400 is the 64.5 mentioned below - a single bad update, not
a plateau; it is at 777 by 204,800.

### The curve is noisy, and that is worth stating

Single updates swing by more than a thousand. After step 100,000, each seed
still has at least one update far below its trend: 579.9 for seed 0 at step
143,360, 150.8 for seed 1 at 126,976, and 64.5 for seed 2 at 102,400. One
update is 4096 environment steps, about four episodes, so a single unlucky
rollout moves the point a long way. Read the trend, not the points.

## Cross-check: the weights, not just the log

`rollout/reward` is measured during collection, with whatever exploration the
sampler adds. To check that the improvement is in the policy rather than in
the logging, the checkpoint from seed 0 at step 245,760 was loaded into the
`eval` algorithm and run for 5 episodes on **environment seeds it had never
seen** (`--runner.seed 100`):

| | episode return |
|---|---|
| Evaluation of the checkpoint | **1339.1** |
| Training-time value at the same step | 1489.5 |

Within about 10%, which is the scale of episode-to-episode variance here. The
learning is in the weights, the checkpoint round-trips, and the `eval` path -
never previously exercised - works.

## What this does and does not support

**Supported:**

* PlugRL trains a policy through its full loop - env client, wire protocol,
  batched inference, feedback channel, learner - and the result is
  reproducible across three seeds on a laptop with no GPU.
* Checkpoints save and load correctly, and a saved policy performs as its
  training log said it did.
* The measured throughput is 70-83 environment steps per second per run with
  three runs sharing 16 cores, and about 140 with one run alone.

**Not supported:**

* Anything about VLAs. This is a 272k-parameter MLP on continuous control.
  The openpi policy path exists in `plugrl-server` and **has never been
  executed**.
* Anything about competitiveness. No baseline was run, no hyperparameter was
  tuned, and HalfCheetah is not a hard benchmark. The claim is "it learns",
  not "it learns well".
* Anything about scale. One environment per client, one client per server,
  one machine. The Ray multi-GPU path cannot run FPO at all - `FPOAlgorithm`
  is not a `DDPAlgorithm`.

## Reproducing

```bash
bash run.sh                    # 3 seeds x 500k steps, about 100 min each
bash run.sh 40000              # a 5-minute version that still shows the trend
python summarise.py            # regenerate summary.tsv and print the curve
```

`run.sh` finds both interpreters itself and checks they can import what they
need before starting; `PLUGRL_SERVER_PYTHON`, `PLUGRL_CLIENT_PYTHON` and
`PLUGRL_ENV_CLIENT` override the guesses. Checkpoints and tensorboard files
are not committed - `summary.tsv` is the result. The server and client logs
are.

## A note on how these runs ended

All three jobs reported a non-zero exit code while their data was complete
and their servers had shut down cleanly. The cause was editing `run.sh` while
it was executing: bash reads a script incrementally, so changing it shifted
the byte offsets under the running interpreter and it resumed at the wrong
place, reporting `unexpected EOF` after the loop had finished. The runs
themselves were unaffected, which the logs and the 123 updates per seed
confirm - but a non-zero exit deserves an explanation rather than a shrug.
