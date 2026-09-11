# E9 measurement protocol (pre-registered)

**Written 2026-09-11, before any access to the hardware this needs.** No data
exists and none can be collected yet, which is the cleanest moment to fix the
rules.

**This file must not be edited after the first data point exists.** If a rule
turns out to be flawed, write a new version saying why the old one was
unusable, and keep both.

---

## The question

Every experiment in this directory so far has run **one** environment client.
E5, E7 and E8 all say so under their limitations. But the architecture's
reason to exist is that one training server can be fed by many environment
clients on many machines - `--num-procs` exists for it, the protocol routes
feedback by environment index for it, and the documentation offers it.

**Nothing has ever tested it.** E9 asks what a second, fourth and eighth
client cost, and whether the server stays correct while they are there.

It also collects, as its one-client cell on two machines, the **L2 rung E7
could not reach**: a physical NIC and a switch rather than a virtual adapter.
E7 named that as its headline limitation and said it needed a second
computer.

## Independent variables

Two, crossed:

| Variable | Levels |
|---|---|
| clients | 1, 2, 4, 8 |
| placement | all clients on the server's machine; all clients on a second machine across a physical LAN |

The 1-client × second-machine cell is E7's L2 rung and is reported as such.

## Held fixed

* the same server configuration throughout, `fpo-policy` with `dummy` or
  `fpo`, stated per run and not varied within a sweep;
* the same environment, same seed policy, same observation payload;
* the same total environment steps across the whole sweep, so a run with 8
  clients does one eighth of the steps each. Throughput is per server, not
  per client, and holding total work fixed is what makes the columns
  comparable;
* one process per client, since `MuJoCoEnv` supports `num_envs=1` only -
  established in E8 and not rediscovered here.

## Metrics

1. **Server throughput** - environment steps per second, across all clients,
   over the whole run excluding the first 30 s.
2. **Per-client latency** - the client's own `infer_wait`, mean and p95.
3. **Correctness, which is the metric that matters most.** Per run:
   * the count of `Feedback for env ... arrived with no step state` warnings,
     which must be **zero**;
   * the count of client reconnects, which must be **zero**;
   * the server's `global_step` against the sum of the clients' reported
     environment steps. These must agree.
4. **Server CPU and memory**, sampled.

## Repetitions and validity

* **3 repetitions** per cell. Median reported, min and max beside it.
* A run counts only if every client exited zero **and** the step accounting
  in metric 3 agrees. A run where the counts disagree is not a slow run, it
  is a wrong one, and it is reported as a correctness failure rather than
  dropped.
* **Ranges that overlap are not a difference.**

## Predictions, recorded before the data

* **P1.** Throughput rises from 1 to 2 clients. *Falsified if* the ranges
  overlap.
* **P2.** Throughput stops rising before 8 clients, because the server
  answers inference in one process and that is the shared resource.
  *Falsified if* 8 clients are faster than 4 by non-overlapping ranges.
* **P3.** Correctness is independent of client count: zero missing-step-state
  warnings and exact step accounting at every level. *Falsified by a single
  occurrence at any level.*
* **P4.** The L2 rung costs more than E7's `win` rung at the same payload.
  *Falsified if* the ranges overlap.

**P3 is the real point.** P1 and P2 are performance curiosity; P3 is whether
the thing the architecture is for actually works. A failure there is the
finding, and it would be a more important result than any timing number in
this directory.

## Known limitations, stated in advance

* **One server process.** Nothing here tests the Ray path, which the
  documentation already marks unsupported because it builds its worker list
  from the local GPU count.
* **Clients are identical.** Real deployments mix fast and slow environments,
  and a slow client holding up a batch is a plausible failure this does not
  probe.
* **No failure injection.** Clients are not killed mid-run. Given that E8
  found a silent corruption on the reconnect path, that is an obvious next
  experiment and is deliberately not folded in here, because mixing a
  scaling question with a fault-tolerance question yields neither answer.
* **Whatever the cluster is.** Node type, interconnect and scheduler will be
  recorded per run and are not controlled.

## Stopping rule

If two machines that can reach each other cannot be obtained, the
second-machine column is recorded as not measured, with the reason, and the
same-machine column is reported alone. That outcome is stated here in advance
so it cannot be presented later as a design choice.

If a cell cannot be measured in 45 minutes, it is recorded as not measured.
No cell is retried more than twice.

## Record format

`results/summary.tsv`, one row per repetition:

```
placement  clients  rep  total_steps  wall_s  steps_per_s  infer_wait_mean_ms  infer_wait_p95_ms  no_step_state_warnings  reconnects  step_accounting_ok  valid
```

`valid` is false for any run whose clients did not all exit zero. Rows with
`step_accounting_ok` false are kept, counted, and reported as correctness
failures rather than excluded.
