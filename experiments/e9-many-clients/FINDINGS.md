# E9: eight clients on one server, and the accounting never slipped once

2026-09-13 · one cluster node, 128 cores, CPU only · FPO on HalfCheetah-v5 ·
12 runs, 12 valid, none skipped

Protocol pre-registered in [`PROTOCOL.md`](PROTOCOL.md), committed before this
project had access to any cluster. Corrections in [`AMENDMENT.md`](AMENDMENT.md).

## In one sentence

The thing the architecture exists for - one training server fed by many
environment clients - works, and works exactly: across 12 runs at 1, 2, 4 and
8 clients, **every run's client step count equalled the server's global step
to the unit**, with zero missing-state warnings and zero reconnects.

## The correctness result, which is the one that matters

The protocol said so in advance, and it is worth repeating before any timing
number:

> **P3 is the real point.** P1 and P2 are performance curiosity; P3 is whether
> the thing the architecture is for actually works. A failure there is the
> finding, and it would be a more important result than any timing number in
> this directory.

**P3 - correctness is independent of client count. CONFIRMED.**

| clients | runs | `no_step_state_warnings` | reconnects | step accounting |
|---|---|---|---|---|
| 1 | 3 | 0 | 0 | exact, 3/3 |
| 2 | 3 | 0 | 0 | exact, 3/3 |
| 4 | 3 | 0 | 0 | exact, 3/3 |
| 8 | 3 | 0 | 0 | exact, 3/3 |

"Exact" means the sum of the clients' own reported `env_steps` equals the
server's `global_step`, unit for unit, in every one of the twelve runs.

That warning is not decoration. It was added to the server days before this
experiment, precisely to make a lost per-environment state audible rather than
silent, and E9 is the first thing to put it under load. It never fired.

## Throughput, and a prediction that did not survive

Median of three runs, `[min, max]` beside it, environment steps per second
across all clients with the total work held fixed at 16384 steps:

| clients | throughput | per client |
|---|---|---|
| 1 | 244.6 [244.6, 252.1] | 244.6 |
| 2 | 292.6 [292.6, 309.2] | 146.3 |
| 4 | 442.9 [442.9, 455.2] | 110.7 |
| 8 | 607.2 [606.9, 630.5] | 75.9 |

**P1 - throughput rises from 1 to 2 clients. CONFIRMED.** The ranges do not
overlap.

**P2 - throughput stops rising before 8 clients. FALSIFIED.** The prediction
named its own falsifier: *"falsified if 8 clients are faster than 4 by
non-overlapping ranges."* They are, and by a wide margin - [442.9, 455.2]
against [606.9, 630.5]. The reasoning behind P2 was that the server answers
inference in one process and that process is the shared resource. It is
shared, and it is not yet saturated at eight clients on this hardware.

Scaling is real but sublinear: **2.48x the throughput for 8x the clients**,
and per-client throughput falls from 244.6 to 75.9 steps per second. The
client-side wait per run tells the same story from the other end, dropping
from 53.8 s to 21.6 s as the fixed 16384 steps are split more ways.

Where the ceiling actually is, this experiment does not say. It says only that
it is above eight.

**P4 - the L2 rung costs more than E7's `win` rung. NOT MEASURED.** The node
available here is a container that can see only itself, so there is no second
machine to place clients on. The protocol's stopping rule required this
outcome be stated rather than presented as a design choice, and E7's headline
limitation therefore still stands.

## One small thing, honestly small

Runs overshoot the 16384 steps the server was asked for, and the overshoot
grows with client count: 1 step at one client, 2-3 at two, 3-4 at four, and
3-10 at eight. Clients that are mid-step when the stop arrives finish that
step. It is a rounding artefact of the stopping protocol, it is bounded by the
client count, and it is reported because it is visible in `total_steps` and a
reader would otherwise wonder.

## What this does and does not support

**Supported:**

* One server serves 1 to 8 concurrent env clients without losing or
  mis-attributing a single environment step.
* Throughput still rises at 8 clients; the inference path is not the binding
  constraint at that scale on this hardware.

**Not supported:**

* **Anything above 8 clients.** The curve was still climbing when the sweep
  ended.
* **Anything across machines.** Every client here shares a kernel with the
  server. E7's machine boundary is not in this measurement, and P4 is
  unmeasured for the same reason.
* **Anything under failure.** No client was killed mid-run. Given that E8
  found a silent corruption on the reconnect path, that is the obvious next
  experiment, and the protocol deliberately kept it out so that a scaling
  question and a fault-tolerance question would not be answered together and
  badly.
* **Anything about a VLA.** The policy here is E6's 272k-parameter MLP. E10
  measured what a real VLA forward costs; nothing has yet put one behind
  several clients at once.

## Reproducing

```bash
bash run.sh                    # 4 client counts x 3 reps, ~12 min
STEPS=2048 BUFFER=512 bash run.sh   # a two-minute version
```

`results/summary.tsv` has one row per run. A run counts only if every client
exited zero **and** its step accounting agrees; a run where the counts
disagree is reported as a correctness failure rather than dropped. All 12
were valid.
