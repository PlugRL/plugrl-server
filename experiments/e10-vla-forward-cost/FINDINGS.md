# E10: the boundary really is cheap, and now it is measured rather than borrowed

2026-09-11 · 8x RTX 3090 cluster node, 128 cores · openpi pi0 built with random
weights · 12 launches, 12 valid, none skipped

Protocol pre-registered in [`PROTOCOL.md`](PROTOCOL.md), committed before this
project had access to any GPU. Deviations in [`AMENDMENT.md`](AMENDMENT.md).

## In one sentence

A VLA forward costs **34.9 ms** for the policy PlugRL actually ships and
**100.0 ms** for the full-size one, against E7's **1.3 ms** to cross a machine
boundary - so the process split costs **1.3% to 3.6% of a step**, and "the
boundary is cheap" stops being a borrowed figure and becomes a measurement.

## The numbers

Median of three independent process launches, `[min, max]` beside it. One
action inference means the whole `sample_actions` call, five denoising steps,
which is what a server answers per `infer`.

| policy | parameters | GPU (bf16) | CPU (fp32) |
|---|---|---|---|
| `pi0-tiny`, what PlugRL ships | 1,710,323,472 | **34.858** [34.614, 35.033] | 3237.8 [3209.0, 3240.7] |
| `pi0-base`, full size | 3,616,757,520 | **99.955** [99.738, 100.007] | 13450.9 [13401.2, 13802.8] |

The GPU spread across three launches is under half a millisecond on a 35 ms
call. That is tighter than anything else in this directory, and it is the one
place a tight number is easy: no network, no scheduler, no other process.

## The predictions, scored

**P1 - the GPU forward exceeds 10 ms at batch 1. CONFIRMED.** 34.9 ms and
100.0 ms.

**P2 - the forward is at least 10x the 1.3 ms cross-machine boundary, so
"single-digit percentage of a step" survives. CONFIRMED**, with room:

| policy | forward / boundary | boundary's share of one step |
|---|---|---|
| `pi0-tiny` | 26.7x | **3.61%** |
| `pi0-base` | 76.7x | **1.29%** |

This was the prediction that mattered. The protocol committed in advance to
withdrawing "cheap" from E5 and E7 if the ratio came back under 10x. It did
not, so the word stays - and it now rests on a number measured here instead
of one taken from the literature.

**P3 - the same forward is more than 10x slower on CPU, so a CPU figure
cannot stand in for this one. CONFIRMED.** 92.9x for the tiny policy, 134.6x
for the full one. Every previous experiment in this directory ran CPU-only;
this is why none of them could have answered the question.

## What the run found that it was not looking for

`PI0Pytorch.__init__` ends with

```python
self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
```

so the first call is not an inference, it is a compilation:

| policy / device | cold cache | warm cache |
|---|---|---|
| tiny / GPU | **317 s** | 13.5 s, 15.5 s |
| base / GPU | **305 s** | 20.3 s, 20.9 s |
| tiny / CPU | 106 s | 20.6 s, 20.6 s |
| base / CPU | 122 s | 41.6 s, 41.0 s |

A training server answering its first `infer` blocks for the whole of that.
The env client's WebSocket keepalive fires at **20 s** and the server's own
`FEEDBACK_WAIT_TIMEOUT` is **60 s**. So a cold start exceeds both by a wide
margin, and even a warm cache exceeds the keepalive for three of the four
configurations.

**This is the mechanism E8 went looking for and did not find.** E8 refuted the
idea that a long learn step drops the connection - learning runs off the event
loop, and 184 s of blocking closed nothing. The VLA's first inference is a
different matter: it happens on the server's request path, and it is minutes.
Nothing here demonstrates a drop, because these runs had no WebSocket in them.
It is a hypothesis with a measured magnitude, which is more than E8's had, and
it should be tested rather than believed.

## What this does and does not support

**Supported:**

* A VLA forward on a current GPU is tens to hundreds of milliseconds at batch
  1, for this policy family, with random weights.
* Against it, the process boundary E5 and E7 measured is a low single-digit
  percentage of a step. The architecture's cost argument holds.
* The openpi policy path executes. The experiments index has said since it was
  written that no VLA had ever run through this system; that is no longer
  true, with the qualification below.

**Not supported:**

* **This is not plugrl-server running a VLA.** `Pi0Policy` cannot be
  constructed without a checkpoint - it asserts one - so the measurement
  builds `PI0Pytorch` directly, the object `load_pytorch` makes before it
  loads weights. The model ran. The server wrapper around it did not.
* **Nothing about trained behaviour.** Random weights measure cost, not
  competence, which is exactly why the protocol committed to them in advance.
* **Nothing about batching.** Batch 1 throughout. Batching improves the
  forward and leaves the per-step boundary unchanged, so batch 1 is the least
  favourable choice for the ratio above, and the ratio still holds.
* **One hardware generation.** An RTX 3090. A faster accelerator shrinks the
  forward and makes the boundary relatively more expensive; how much is
  unmeasured.
* **The compile-timeout interaction is untested.** See above.

## Reproducing

```bash
bash run.sh          # 12 launches: 2 policies x 2 devices x 3 launches
```

Needs a GPU, the openpi sources, and roughly 40 minutes. `results/summary.tsv`
has one row per launch including a `valid` column; a launch counts only if the
action came back with the shape the config declares. All 12 were valid and
`results/not-measured.txt` is empty.
