# E5: the boundary costs under a millisecond, and the default setting was wasting more than that

2026-09-09 · WSL2 Ubuntu 22.04 · loopback · plugrl-server @ 4aaebe0
Client: `e2-cross-language/plugrl_client.cpp` (C++, no third-party libraries)

## In one sentence

Sending a 184 KiB observation across the process boundary costs **under a
millisecond** (0.5-0.7 ms). A naive measurement says 1.65 ms, but about
1.1 ms of that is the server's scheduler waiting to poll - a tunable
constant, not an architectural cost.

## Payload size sweep

The server's artificial delay is off (`--algo.fake-inference-duration-sec 0`),
so these numbers contain only serialization, transport, deserialization and
an empty policy forward.

| Configuration | Uplink payload | pack | RTT | unpack |
|---|---|---|---|---|
| 1 camera @ 128px | 48 KiB | 0.022 ms | 1.709 ms | 0.004 ms |
| 1 camera @ 224px | 147 KiB | 0.124 ms | 1.680 ms | 0.005 ms |
| **2 cameras @ 224px (dummy-v1)** | **184 KiB** | **0.122 ms** | **1.603 ms** | **0.003 ms** |
| 1 camera @ 448px | 588 KiB | 0.575 ms | 1.796 ms | 0.005 ms |
| 2 cameras @ 224px, batch 4 | 735 KiB | 0.241 ms | 1.955 ms | 0.005 ms |
| 2 cameras @ 224px, batch 16 | 2942 KiB | 3.080 ms | 5.622 ms | 0.007 ms |

What that says:

* **RTT barely moves from 48 KiB to 588 KiB** (1.68 to 1.80 ms). At this
  scale the dominant term is fixed overhead, not payload.
* **Packing stays sub-millisecond**, reaching 0.575 ms only at 588 KiB.
* **Unpacking is negligible** (~0.005 ms) - the downlink is just an action.
* Only at 2.9 MiB (batch 16) does anything climb noticeably.

## Splitting the fixed overhead: how much is polling?

Payload fixed at 184 KiB, varying only the server's
`SCHEDULER_SLEEP_INTERVAL`:

| Poll interval | mean RTT (run 1) | mean RTT (run 2) |
|---|---|---|
| **1.000 ms (repository default)** | **1.653 ms** | **1.570 ms** |
| 0.500 ms | 1.134 ms | 1.503 ms |
| 0.100 ms | 0.532 ms | 0.701 ms |
| 0.010 ms | 0.556 ms | (client failed; discarded) |

The 1 ms setting is stable across runs, within 5%. The 0.1 ms setting is
visibly noisy (0.532 / 0.701 ms; a separate end-to-end check gave mean 1.060,
median 0.883 ms). So:

| Component | Cost |
|---|---|
| serialization + loopback transport + empty policy forward | **0.5-0.7 ms** |
| scheduler poll wait (at the default) | **~0.9-1.1 ms** |

**These RTT numbers establish an order of magnitude, not a ratio.** The
quantitative claim is the throughput median over five repetitions, below.

## Two conclusions

### 1. The boundary is cheap

A VLA policy forward is tens of milliseconds. Against that, **moving a
two-camera 224px observation across the boundary costs about 0.5 ms** - a
single-digit percentage. That is what makes "run the environment on any
machine" defensible on latency grounds.

The honest limit: **this is loopback, which is a lower bound.** Real
cross-machine numbers will be higher. What that adds is network round trip,
and the serialization cost - the part PlugRL controls - is measured here and
is small.

Bandwidth, for planning: 184 KiB/step x 30 Hz is about 5.4 MB/s per env
client, so a gigabit link saturates at roughly 22 clients.

### 2. The default setting cost about 1.1 ms per step, and removing it cost nothing

`SCHEDULER_SLEEP_INTERVAL = 0.001` in `websocket_agent_server.py` made every
inference request wait about 1.1 ms longer on average. At `0.0001`:

| Poll interval | Throughput (median of 5) | CPU per exchange | Idle CPU | RTT scale |
|---|---|---|---|---|
| **1.000 ms (old default)** | 457/s (425-470) | 0.890 ms | 7.3% | ~1.6 ms |
| **0.100 ms (adopted)** | **803/s** (692-892) | 0.758 ms | 6.1% | ~0.5-1.0 ms |
| 0.010 ms | 861/s (657-874) | 0.710 ms | 6.2% | not repeated enough |

This change has no downside that the data can find:

* **1.76x throughput**, and the ranges do not overlap (425-470 vs 692-892).
* **15% less CPU per exchange** - tighter polling is not spinning, it is
  waiting less.
* **Idle CPU unchanged** (7.3% to 6.1%, within noise).

The 0.1 ms and 0.01 ms ranges do overlap (692-892 vs 657-874), so **this data
cannot separate them**, and the more conservative 0.1 ms was adopted.

At 200 steps per episode that is 0.22 s saved per episode, and more for
high-frequency control.

#### Three corrections to earlier versions of this document

1. It said "lowering this will raise idle CPU; confirm by measurement."
   **It was measured, and it does not.** Idle usage is the same across all
   three settings (6.1%-7.3%). Writing an unverified worry as a warning was
   wrong.
2. It said, from a single sample, that 0.01 ms gave no further benefit and
   was possibly worse. **That was noise**: over five repetitions its median
   throughput is slightly *higher* than 0.1 ms. One sample is not enough to
   justify a production change.
3. It claimed a **3.1x RTT improvement** from one pair of samples. **That
   does not survive remeasurement** - the 1 ms setting is stable at
   1.57-1.65 ms, but the 0.1 ms setting ranges over 0.53-1.06 ms, and the
   sample is too small for a ratio. The defensible number is the 1.76x
   throughput over five repetitions.

#### A harness defect, fixed

While repeating the runs, the measurement script was found to treat **failed
runs as valid data** - it reported 118879 exchanges/s from a client that had
failed to connect. `run-cpu-per-exchange.sh` now requires the client log to
confirm it completed the requested number of exchanges, and discards the run
otherwise. The n column above is valid samples only.

## One configuration that could not be measured

The `states only (no cameras)` case failed with `server returned an empty
action`. That is the dummy policy defect recorded in E2: `_infer_batch_size`
takes `next(iter(obs.values()))` and then `len()`, which computes batch=0
when `images` is an empty dict, and **returns an empty action silently
instead of raising**. A proprioception-only robot with no cameras lands on it
exactly.

## Reproducing

```bash
wsl bash run.sh                    # payload size sweep
wsl bash run-isolate-poll.sh       # isolate the polling overhead
wsl bash run-idle-cost.sh          # idle and busy CPU usage
wsl bash repeat.sh                 # 5 repetitions + medians, with validation
```

* `server_with_interval.py` starts a server with a given poll interval.
* Results are in `results/summary.tsv`.
