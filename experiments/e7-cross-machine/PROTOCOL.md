# E7 measurement protocol (pre-registered)

**Written 2026-09-11, before any data was collected.** Committed before the
first run, which is the only thing that makes a pre-registration worth
anything.

**This file must not be edited after the first data point exists.** If a rule
turns out to be flawed, write a new version saying why the old one was
unusable, and keep both.

---

## The question

E5 measured the process boundary at 0.5-0.7 ms for a 184 KiB observation and
said plainly that this is **loopback, and therefore a lower bound**. The
architecture's whole cost argument - that putting the environment on another
machine is cheap relative to a VLA forward pass - rests on a number that has
never been measured across a real network stack.

**E7 measures what the boundary costs when packets leave the loopback
interface.**

## Independent variable

The network path, and nothing else:

| Rung | Path | Available |
|---|---|---|
| L0 | loopback, `127.0.0.1` | yes |
| L1 | WSL2 guest to Windows host, across the virtual Ethernet adapter and NAT | yes |
| L2 | two machines on one LAN, physical NIC and switch | only if a second machine is provided |
| L3 | wide area, e.g. a cloud VM | not planned |

L0 and L1 will be measured. L2 is reported if and only if a second machine
becomes available; its absence is a stated limitation, not a silent gap.

## Held fixed

* the same client binary, `plugrl-protocol/examples/plugrl_client.cpp`, built
  once and reused;
* the same server, `examples/conformance_server.py`, which runs no policy, so
  the number is transport and serialization only;
* the same payload for the headline comparison: batch 1, two cameras at
  224px, 184 KiB uplink;
* the same scheduler interval - the conformance server has none, which is
  deliberate: E5 showed the shipped 1 ms poll dominated the measurement, and
  including it here would measure the scheduler again rather than the
  network.

## Metrics

Reported per run by the client itself:

1. **RTT** - mean, p50, p95 over the samples after a 3-exchange warm-up.
2. **pack** and **unpack** time, same statistics.
3. **exchanges per second** over the whole run.

**RTT is reported for scale; throughput is the quantitative claim.** This is
E5's lesson, learned the hard way: RTT across repetitions of the same
configuration varied by a factor of two there, and a ratio taken from one
pair of samples did not survive remeasurement. Any ratio in E7 comes from
medians over repetitions with their ranges shown, or it does not appear.

## Repetitions and validity

* **5 repetitions** of every (rung x payload) cell. The median is the
  reported value; the min and max are reported beside it.
* A run counts only if the client's own output confirms it completed the
  requested number of exchanges. E5's harness once reported 118879
  exchanges/s from a client that had failed to connect; the same trap is
  explicitly guarded here.
* **Ranges that overlap are not a difference.** If two cells' [min, max]
  overlap, the finding is "this data cannot separate them", and it is
  reported that way.

## Payload sweep

At each rung, three payloads: 48 KiB (one 128px camera), 184 KiB (two 224px
cameras, the headline), and 588 KiB (one 448px camera). Same sizes as E5, so
the two are directly comparable.

## Predictions, recorded before the data

Stating these now so they can be wrong in public.

* **P1.** L1 RTT is higher than L0 RTT at the headline payload. *Falsified
  if* the L1 and L0 medians' ranges overlap.
* **P2.** Payload size matters more at L1 than at L0. On loopback E5 found
  RTT almost flat from 48 KiB to 588 KiB - fixed overhead dominated. Across a
  NIC, bandwidth should start to bind, so the 588/48 RTT ratio should be
  larger at L1 than at L0. *Falsified if* that ratio is the same or smaller
  at L1.
* **P3.** Pack and unpack times are unchanged across rungs, because they
  happen before the bytes reach the network. *Falsified if* they move by more
  than the repetition spread.

P3 is a control: if it fails, the measurement is contaminated by something
other than the network - most likely CPU contention between guest and host.

## Known limitations, stated in advance

* **L1 is not two machines.** The WSL2 guest and the Windows host share one
  CPU and one memory bus. The virtual adapter is a real network stack, but
  the packets never touch physical hardware. L1 is an intermediate point
  between loopback and a LAN, and will be described that way, not as a
  cross-machine number.
* **CPU contention is uncontrolled.** Both ends run on the same machine at
  L0 and L1. P3 exists to detect this.
* **The server runs no policy.** A real run adds a policy forward, which for
  a VLA is tens of milliseconds. That is the comparison that matters, and it
  makes the boundary look *better*, so leaving it out is the conservative
  choice.
* **One client implementation.** The C++ client is used throughout because it
  has no third-party libraries and therefore no library-version variable.
  Nothing here claims the Python client would measure the same.

## Stopping rule

If a cell cannot be measured in 30 minutes, it is recorded as not measured
with the reason, and the sweep continues. No cell is retried more than twice.

## Record format

`results/summary.tsv`, one row per repetition:

```
rung  payload_kib  rep  exchanges  rtt_mean_ms  rtt_p50_ms  rtt_p95_ms  pack_ms  unpack_ms  exchanges_per_s  valid
```

`valid` is false for any run whose client did not confirm completion; invalid
rows are kept in the file and excluded from every statistic.
