# E10 measurement protocol (pre-registered)

**Written 2026-09-11, before any access to the hardware this needs.** There
is no data, and no way to get any yet, which makes this the cleanest moment
to write it down.

**This file must not be edited after the first data point exists.** If a rule
turns out to be flawed, write a new version saying why the old one was
unusable, and keep both.

---

## The question

E5 concluded that the process boundary is **cheap**. The measurement behind
that word is 0.5 ms on loopback, and E7 raised it to 1.3 ms across a machine
boundary. Neither is the claim. The claim is a ratio, and its denominator -
"a VLA policy forward is tens of milliseconds" - **has never been measured by
anything in this project**. It is borrowed from the literature and it is
load-bearing: at a 30 ms forward the boundary is under 2% of a step, at a
2 ms forward it is a fifth of one, and the architecture's cost argument
inverts.

**E10 measures the denominator.**

A second question follows for free, and is the one the experiments index has
listed under "What is missing" since it was written: **no VLA has ever run
through this system.** The policy path exists in
`plugrl_server/policy/openpi/` and has never been executed.

## Independent variable

The policy, and nothing else:

| Policy | Parameters | Available |
|---|---|---|
| `fpo-policy` (default) | 272,423 | yes, measured in E6 |
| `pi0` randomly initialised | ~3B | needs the `openpi` extra |
| `pi0` from a released checkpoint | ~3B | needs the extra and the download |

The randomly initialised case is the one this protocol commits to. **Weights
do not change the cost of a forward pass**, only its output, so a checkpoint
is not required to answer the question - and saying so in advance stops a
failed download from being reported as a failed experiment.

## Held fixed

* one observation shape throughout: batch 1, two cameras at 224 px, the same
  184 KiB payload E5 and E7 used as their headline, so the three are directly
  comparable;
* the same action horizon;
* inference only. No learn step, no optimiser, no backward pass.
* `torch.inference_mode()`, and the device stated with every number.

## Metrics

1. **Forward latency** - mean, p50, p95 over 30 calls after 5 warm-up calls.
   Warm-up is not optional on a GPU and the discarded calls are reported.
2. **The ratio that matters**: boundary cost over forward cost, computed from
   E7's 1.3 ms cross-machine figure and this experiment's median, with the
   range each carries.
3. **Device, precision and memory**, recorded per run. A number without them
   is not a number.

## Repetitions and validity

* **3 independent process launches** per cell, not 3 loops inside one
  process. A single process measures one allocator state and one set of
  kernels already resident.
* A run counts only if the policy returned an action of the expected shape
  and dtype. E5's harness once reported a throughput figure from a client
  that had failed to connect; the same trap is guarded here.
* **Ranges that overlap are not a difference.**

## Predictions, recorded before the data

Stating these so they can be wrong in public.

* **P1.** On a GPU, the pi0 forward exceeds 10 ms at batch 1. *Falsified if*
  the median is below 10 ms.
* **P2.** The forward is at least 10x the 1.3 ms cross-machine boundary
  cost, so "the boundary is a single-digit percentage of a step" survives.
  *Falsified if* the ratio is under 10x.
* **P3.** On CPU the same forward is more than 10x slower than on GPU, which
  is why a CPU-only number cannot stand in for this measurement. *Falsified
  if* the CPU/GPU ratio is under 10x.

P2 is the one that matters. If it fails, E5's and E7's "cheap" has to be
withdrawn, and this document exists to make that outcome reportable rather
than avoidable.

## Known limitations, stated in advance

* **One policy family.** pi0 is not every VLA. A smaller one will be faster
  and the ratio will be worse; this measures the case the project actually
  cites.
* **Random weights.** Justified above. It is still a difference from a real
  deployment and is stated on every number.
* **Batch 1.** Real serving batches. Batching improves the forward's
  throughput and leaves the per-step boundary cost unchanged, so batch 1 is
  the conservative choice for the ratio and the least favourable for PlugRL.
* **No end-to-end training run.** Executing the forward is not the same as
  training a VLA through this system. That remains unmeasured and this
  protocol does not claim it.

## Stopping rule

If the `openpi` extra cannot be installed, or the policy cannot be
constructed, that is recorded as the finding with the error, and the
experiment stops. A blocked path is a result about the path.

If a cell cannot be measured in 30 minutes, it is recorded as not measured
with the reason. No cell is retried more than twice.

## Record format

`results/summary.tsv`, one row per launch:

```
policy  device  dtype  params  batch  launch  warmup_calls  n  fwd_mean_ms  fwd_p50_ms  fwd_p95_ms  peak_mem_mb  action_shape  valid
```

`valid` is false for any launch whose action shape or dtype was wrong;
invalid rows are kept in the file and excluded from every statistic.
