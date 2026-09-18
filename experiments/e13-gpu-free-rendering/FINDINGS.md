# E13: rendering without a GPU costs 1.91x, not nothing

**LIBERO renders without a GPU, and the loop keeps working: ten clients, 30
episodes, all successful either way. It is not free. The prediction registered
here was that a queue of ten clients would hide the penalty and keep wall
clock under 1.3x. It came out at 1.91x, and that prediction is falsified. What
is true is narrower and still useful: an 11x penalty per rendered step becomes
1.91x in the loop, against 3.07x with a single client.**

Pre-registered in [`PROTOCOL.md`](PROTOCOL.md) on 2026-09-18, with four
exploratory measurements declared in it rather than predicted.

## What ran

| | |
|---|---|
| Server | `pi0-policy` / `pi05_libero` / `eval`, CUDA, one GPU, unchanged between cells |
| Client install | E12's CUDA-free environment, identical in both cells |
| Clients | 10 processes, 3 episodes each, `libero_spatial` task 0, states in order, seed 7 |
| Renderers | NVIDIA EGL against Mesa `osmesa`, software |
| `libosmesa6` | not installed on the cluster; extracted from the Ubuntu 22.04 package into `$R/opt/osmesa` and reached through `LD_LIBRARY_PATH`. The shared container was not modified |
| Machine | cluster `qz103`. Both cells ran on GPU 1 in one ten-minute window, at load 16-24, after waiting for a card to free |

## The measurement

| | `ten-egl` | `ten-osmesa` | ratio |
|---|---|---|---|
| **wall clock** | **133 s** | **254 s** | **1.91x** |
| episodes | 30 | 30 | |
| successes | **30** | **30** | |
| mean collect time | 96.7 s | 202.6 s | 2.09x |
| mean env step time | 7.8 s | 80.6 s | **10.3x** |
| mean infer wait | 88.9 s | 121.9 s | 1.37x |
| mean env step share | 0.081 | 0.398 | |

Per process in [`results/summary.tsv`](results/summary.tsv) and
[`results/queue.log`](results/queue.log).

## The prediction that failed

The protocol registered: *"`ten-osmesa` wall clock is less than 1.3x
`ten-egl`'s - that is, the 11x stepping penalty is mostly hidden by the queue.
Falsified at 1.3x or above."*

**1.91x. Falsified.**

The reasoning behind it was that with ten clients sharing one policy, each
client spends most of its time queued, so slower stepping would fit inside
that wait. The first half held: stepping is 10.3x slower and the loop is only
1.91x slower, so most of the penalty is absorbed. The second half did not:
enough of it is left to nearly double the run.

## Why, and a second effect the protocol did not anticipate

Stepping does not simply hide behind the wait, because **the wait itself grew
by 1.37x**. Per inference call that is about 1.48 s against 2.03 s.

Nothing about the server changed between cells, so the likely mechanism is on
the client side: slower clients issue requests less densely, the scheduler
fires on smaller batches, and each batch's fixed cost is spread over fewer
requests. That is consistent with E9, which found throughput still rising at
eight clients - this loop is in the regime where more simultaneous requests
make the server more efficient, and software rendering takes it the other way.

**This is an explanation, not a measurement.** Batch sizes were not recorded.
A cell that varies the client count with the renderer held fixed would test
it, and is not run here.

## What holds

- **The environment side runs with no GPU.** Ten clients, 30 episodes, every
  one successful, rendering entirely on the CPU.
- **Success is unaffected**: 30 of 30 in both cells. With an unseeded policy
  exact equality of trajectories is not available - step counts differ, 2,799
  against 2,841 - which is why the protocol asked for a range and not equality.
- **The penalty shrinks with concurrency**: 11x isolated, 3.07x for one client
  in the loop, 1.91x for ten. It does not vanish.
- Combined with [E12](../e12-cuda-free-rollout/FINDINGS.md), a LIBERO rollout
  source is **3.4G with no CUDA and no GPU**, at 1.91x the wall clock of one
  that has both.

## What this does not support

- **Not "software rendering is free".** It nearly doubles the run. Whether
  that is worth it depends on what the alternative machine costs.
- **Not a claim about small machines.** This is a 128-core server rendering
  ten environments at once. A robot's onboard computer with four cores would
  not behave like this, and nothing here measures that.
- **Not a pixel comparison.** Software and hardware rasterisers need not
  produce identical frames. Only timing and task success were compared; the
  success counts being equal is evidence the images are good enough for this
  policy, not that they are the same.
- **Not cross-machine.** Server and clients were on one machine. E7 measured
  the cost of leaving it, at +0.52 ms per 184 KiB observation, and that is not
  re-measured here.
- **One task, three episodes per client.** A timing result, not a success-rate
  result.

## Predictions, against the outcome

| | prediction | outcome |
|---|---|---|
| 1 | both cells complete 30 episodes, no client failing | **confirmed** |
| 2 | `ten-osmesa` wall clock under 1.3x | **falsified** - 1.91x |
| 3 | `env_step_frac` below 0.5 for the median client | **confirmed** - 0.398 |
| 4 | success counts within 3 of each other | **confirmed** - 30 and 30 |

## Budget

Two cells, 133 s and 254 s of client time, plus about 90 s of model loading
each. The queue waited ten minutes for a card: the first attempt at `ten-egl`
had died with `CUDA out of memory` beside an unrelated job holding 15.5G on
every GPU, and the protocol's stopping rule is to wait rather than compare
against a differently loaded machine.
