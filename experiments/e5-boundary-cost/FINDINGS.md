# E5: the boundary costs under a millisecond, and the default setting was wasting about as much again

2026-09-09 · WSL2 Ubuntu 22.04 · loopback · plugrl-server @ 4aaebe0
Client: `plugrl-protocol/examples/plugrl_client.cpp` (C++, no third-party
libraries). The binary these numbers came from was built from an untracked
working copy of that client which is in no repository (693 lines against the
tracked 843); the tracked file takes the same command line and prints the same
TSV, so `run.sh` builds it, but a rerun is a rerun and not a replay.

## In one sentence

Sending a 184 KiB observation across the process boundary costs **under a
millisecond**: mean 0.918 ms, median 0.744 ms over 1997 samples with the
server polling at 0.01 ms (`results/perex-client-0.00001.log`). The same
client against the 1 ms repository default measures mean 1.590 ms, median
1.674 ms (`results/perex-client-0.001.log`). The difference - 0.67 ms on the
mean, 0.93 ms on the median - is the server's scheduler waiting to poll: a
tunable constant, not an architectural cost.

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

* **RTT barely moves from 48 KiB to 588 KiB** (1.71 to 1.80 ms; the whole
  48-588 KiB band is 1.60-1.80 ms). At this scale the dominant term is fixed
  overhead, not payload.
* **Packing stays sub-millisecond**, reaching 0.575 ms only at 588 KiB.
* **Unpacking is negligible** (~0.005 ms) - the downlink is just an action.
* Only at 2.9 MiB (batch 16) does anything climb noticeably.

## Splitting the fixed overhead: how much is polling?

Payload fixed at 184 KiB, varying only the server's
`SCHEDULER_SLEEP_INTERVAL`. Two runs of `run-cpu-per-exchange.sh`, same
client, same payload, 1997 timed samples each:

| Poll interval | mean RTT | p50 RTT | source |
|---|---|---|---|
| **1.000 ms (repository default)** | **1.590 ms** | 1.674 ms | `results/perex-client-0.001.log` |
| 0.010 ms | 0.918 ms | 0.744 ms | `results/perex-client-0.00001.log` |

At a 0.01 ms poll interval the expected poll wait is about 0.005 ms, so that
row is close to the boundary on its own. Differencing the two:

| Component | Cost |
|---|---|
| serialization + loopback transport + empty policy forward | **0.92 ms mean, 0.74 ms median** |
| scheduler poll wait (at the 1 ms default) | **0.67 ms mean, 0.93 ms median** |

Earlier versions of this document put a four-row sweep here (1.000 / 0.500 /
0.100 / 0.010 ms, two 200-step runs each) plus an end-to-end cross-check.
**None of it is reproducible from what is committed** - `run-isolate-poll.sh`
prints the client's TSV to the terminal and never writes it. Those rows have
been removed and are recorded in correction 4; the two rows above replace
them.

**This is one run per setting, not a repeated measurement, and it establishes
an order of magnitude rather than a ratio.** The quantitative claim is the
throughput median over repeated runs, below.

## Two conclusions

### 1. The boundary is cheap

**Moving a two-camera 224px observation across the boundary costs about
0.9 ms on the mean and 0.7 ms on the median** - 1997 samples at the tightest
poll interval, `results/perex-client-0.00001.log`. That is the measurement.
Whether it is cheap depends on what it is being compared against, and the
comparison is not measured here.

> **The assumption this conclusion rests on.** "Cheap" means cheap relative
> to a VLA policy forward in the tens of milliseconds. PlugRL has never run a
> VLA - the experiments index says so under "What is missing" - so that
> number is borrowed from the literature and not established by anything in
> this repository. It is load-bearing: at a 30 ms forward the boundary is
> under 2% of the step and the architecture is defensible on latency grounds;
> at a 2 ms forward it is a fifth of the step and the conclusion flips. The
> measurement above stands either way. The word "cheap" does not.

The other honest limit: **this is loopback, which is a lower bound.** Real
cross-machine numbers will be higher. What that adds is network round trip,
and the serialization cost - the part PlugRL controls - is measured here and
is small. E7 later put a number on the difference: **+0.52 ms** once the
bytes leave the machine.

Bandwidth, for planning: 184 KiB/step x 30 Hz is about 5.4 MiB/s per env
client, so a gigabit link saturates at roughly 22 clients.

### 2. The default setting cost about 0.7-0.9 ms per step, and removing it cost nothing

`SCHEDULER_SLEEP_INTERVAL = 0.001` in `websocket_agent_server.py` made every
inference request wait roughly 0.7-0.9 ms longer on average. At `0.0001`:

| Poll interval | n | Throughput (median of valid runs) | CPU per exchange | Idle CPU | RTT at this setting |
|---|---|---|---|---|---|
| **1.000 ms (old default)** | 5 | 457/s (425-470) | 0.890 ms | 7.3% (uncaptured) | 1.590 ms mean / 1.674 p50 |
| **0.100 ms (adopted)** | 4 | **803/s** (692-892) | 0.758 ms | 6.1% (uncaptured) | not captured |
| 0.010 ms | 3 | 861/s (657-874) | 0.710 ms | 6.2% (uncaptured) | 0.918 ms mean / 0.744 p50 |

`n` is the number of runs that completed all 2000 requested exchanges, counted
from `results/perex.tsv`: 12 rows, five at 1.000 ms, four at 0.100 ms, three
at 0.010 ms. `repeat.sh` prints this column and earlier versions of this table
dropped it, which is how "median of 5" came to stand over a median of four and
a median of three. The idle-CPU column is marked uncaptured because it is
**not reproducible from what is committed**; see correction 4.

This change has no downside that the data can find:

* **1.76x throughput** (457 -> 803/s), and the ranges do not overlap
  (425-470 vs 692-892).
* **15% less CPU per exchange** - tighter polling is not spinning, it is
  waiting less. Same rows, same `n`: 0.890 ms over five runs against 0.758 ms
  over four.
* **Idle CPU unchanged** (7.3% to 6.1%, within noise) - on terminal output
  that was never written to a file. See correction 4.

The 0.1 ms and 0.01 ms ranges do overlap (692-892 vs 657-874), so **this data
cannot separate them**, and the more conservative 0.1 ms was adopted.

At 200 steps per episode the throughput medians give 0.19 s saved per episode
(200/457 = 0.437 s against 200/803 = 0.249 s), and more for high-frequency
control. The 0.22 s printed here before was 200 x the 1.1 ms poll estimate
that correction 4 withdraws.

#### Four corrections to earlier versions of this document

1. It said "lowering this will raise idle CPU; confirm by measurement."
   **It was measured, and it does not.** Idle usage is the same across all
   three settings (6.1%-7.3%). Writing an unverified worry as a warning was
   wrong. Read this together with correction 4: the measurement was run, but
   its output was never written to a file, so the retraction now rests on
   terminal output that cannot be checked.
2. It said, from a single sample, that 0.01 ms gave no further benefit and
   was possibly worse. **That was noise**: over its three valid repetitions
   its median throughput (861/s) is slightly *higher* than the 0.1 ms
   setting's median over four (803/s). One sample is not enough to justify a
   production change. This correction used to say "over five repetitions";
   the 0.01 ms setting has three valid runs, not five.
3. It claimed a **3.1x RTT improvement** from one pair of samples. **That
   does not survive** - two samples cannot carry a ratio, and the pair was
   never captured to a file (correction 4). The defensible number is the
   1.76x throughput, over five valid runs at 1 ms and four at 0.1 ms.
4. Two sets of numbers in earlier versions of this document have **no
   committed file behind them**. Nothing below is a remeasurement; this
   correction only records what lost its backing.
   * The four-row poll-interval RTT sweep - 1.000 ms: 1.653 / 1.570 ms;
     0.500 ms: 1.134 / 1.503 ms; 0.100 ms: 0.532 / 0.701 ms; 0.010 ms:
     0.556 ms - and the end-to-end cross-check quoted as mean 1.060, median
     0.883 ms. `run-isolate-poll.sh` prints the client's TSV fields to the
     terminal and never persists them; `results/poll-0.001.log`,
     `poll-0.0005.log` and `poll-0.0001.log` hold only the server's startup
     and shutdown lines, and `poll-0.00001.log` is empty. **Those rows have
     been removed.** The 1997-sample pair from `run-cpu-per-exchange.sh`,
     which is committed, now carries the polling decomposition and the
     headline.
   * The idle-CPU figures 7.3% / 6.1% / 6.2%. `run-idle-cost.sh` computes the
     percentage into a shell local and prints it; the five `results/idle-*.log`
     files it writes are the server's own stdout and contain no CPU figure.
     **These are kept but flagged as uncaptured**, because correction 1
     retracts a published warning on their strength and deleting them would
     leave that retraction unexplained.

   Both scripts should append their rows to a TSV under `results/` the way
   `run-cpu-per-exchange.sh` does. Until they do, treat the idle-CPU claim as
   unverified and the removed RTT rows as gone.

#### A harness defect, fixed

While repeating the runs, the measurement script was found to treat **failed
runs as valid data** - it reported 118879 exchanges/s from a client that had
failed to connect. `run-cpu-per-exchange.sh` now requires the client log to
confirm it completed the requested number of exchanges, and discards the run
otherwise. Every median above is over valid runs only.

## One configuration that could not be measured

The `states only (no cameras)` case failed with `server returned an empty
action`. That was the dummy policy defect recorded in E2: `_infer_batch_size`
took `next(iter(obs.values()))` and then `len()`, which computed batch=0
when `images` was an empty dict, and **returned an empty action silently
instead of raising**. A proprioception-only robot with no cameras landed on it
exactly.

That defect has since been fixed, after the `4aaebe0` version this document
pins: `_infer_batch_size` in `src/plugrl_server/policy/dummy_policy.py` now
reads `if not obs: return 1` and otherwise defers to `numpy_tree_batch_size`.
The failure recorded above was real when it was measured; the configuration
has not been remeasured since the fix.

## Reproducing

Prerequisite: every script here drives the server venv that E2 builds. Run
`bash ../e2-cross-language/setup-server.sh` first - it creates
`$HOME/.e2-server`, and these scripts call
`$HOME/.e2-server/bin/plugrl-run-server` and `$HOME/.e2-server/bin/python`
directly.

```bash
wsl bash run.sh                    # builds the client, then the payload sweep
wsl bash run-isolate-poll.sh       # isolate the polling overhead
wsl bash run-idle-cost.sh          # idle and busy CPU usage
wsl bash repeat.sh                 # 5 repetitions + medians, with validation
```

* `run.sh` compiles `plugrl-protocol/examples/plugrl_client.cpp` to
  `$HOME/.e5-client`; the other three scripts need that binary to exist
  already, so run it first.
* `server_with_interval.py` starts a server with a given poll interval.
* Results are in `results/summary.tsv` (payload sweep) and `results/perex.tsv`
  (throughput and CPU per exchange, one row per valid run).
  `run-isolate-poll.sh` and `run-idle-cost.sh` write nothing but server logs -
  their numbers go to the terminal only, which is what correction 4 is about.
