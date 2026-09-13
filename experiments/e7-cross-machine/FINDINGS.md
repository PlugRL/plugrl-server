# E7: the boundary costs what leaving the machine costs, not what the network stack costs

2026-09-11 · Windows 11 host, WSL2 Ubuntu 22.04 guest · C++ client from
`plugrl-protocol/examples`, one binary throughout · 45 runs, 45 valid

Protocol pre-registered in [`PROTOCOL.md`](PROTOCOL.md) and committed at
`c5e22cf`, before the first data point.

## In one sentence

Going through a real network stack to the same machine costs **nothing
measurable**; leaving the machine costs **+0.52 ms** on a 184 KiB
observation. E5's loopback number was not hiding the cost of the network
stack - it was hiding the cost of the machine boundary, and those are
different things.

## The decomposition

E5 measured loopback and said it was a lower bound. The obvious next
measurement - loopback versus another machine - cannot be done with one
machine, and the naive substitute is misleading: moving the server from the
WSL2 guest to the Windows host changes the network path *and* the operating
system *and* the Python interpreter at once.

So three rungs, with the client identical throughout:

| rung | path | what it adds |
|---|---|---|
| `lo` | WSL2 client → WSL2 server, `127.0.0.1` | baseline |
| `self` | WSL2 client → WSL2 server, via its own `eth0` address | the network stack, OS held fixed |
| `win` | WSL2 client → Windows server, via the host address | crossing the VM boundary as well |

RTT, median of 5 repetitions, `[min, max]` beside it. Each repetition
contributes the client's *mean* RTT over its 1000 exchanges, so these are
medians of five means - see deviation 3:

| payload | `lo` | `self` | `win` |
|---|---|---|---|
| 48 KiB | 0.585 [0.490, 0.752] | 0.620 [0.547, 0.647] | 0.949 [0.910, 0.987] |
| **184 KiB** | **0.829 [0.761, 0.898]** | **0.782 [0.736, 0.799]** | **1.304 [1.220, 1.359]** |
| 588 KiB | 1.900 [1.455, 1.990] | 1.805 [1.657, 2.196] | 3.218 [2.158, 3.866] |

Read across each row:

* **`lo` → `self` is not a difference.** At every payload the ranges overlap,
  and at two of three the median moves the *wrong* way. Addressing the host
  by its external IP instead of loopback changes nothing that this
  measurement can see - the kernel short-circuits traffic to a local
  interface, and the "network stack" is not where the cost is.
* **`self` → `win` is a difference**, and at the headline payload the ranges
  do not come close to touching: 0.782 [0.736, 0.799] against
  1.304 [1.220, 1.359]. **+0.52 ms, all of it from leaving the virtual
  machine.**

At 588 KiB the `self`→`win` ranges do overlap ([1.657, 2.196] against
[2.158, 3.866]), so that row is reported as a difference in medians that
this data cannot establish. The overlap is one sample wide; more repetitions
would probably separate it, and the protocol's rule is that overlap means
"cannot separate", not "separated if you squint".

## The predictions, scored

**P1 - `win` RTT exceeds `lo` RTT at 184 KiB. CONFIRMED.**
0.829 [0.761, 0.898] against 1.304 [1.220, 1.359]; the ranges do not
overlap. The ratio is 1.57x.

**P2 - payload matters more across a NIC than on loopback. NOT SUPPORTED.**
The pre-registered test was the 588/48 RTT ratio per rung:

| rung | median ratio | range the ratio could take |
|---|---|---|
| `lo` | 3.25 | [1.93, 4.06] |
| `self` | 2.91 | [2.56, 4.01] |
| `win` | 3.39 | [2.19, 4.25] |

The medians order as predicted and the ranges overlap almost completely.
By the protocol's own rule this is "cannot separate", and the prediction
does not get credit for the ordering.

**P3 - pack and unpack do not move with the network path. CONFIRMED.**

| payload | pack `lo`/`self`/`win` | unpack `lo`/`self`/`win` |
|---|---|---|
| 48 KiB | 0.034 / 0.037 / 0.040 | 0.011 / 0.011 / 0.008 |
| 184 KiB | 0.249 / 0.236 / 0.235 | 0.013 / 0.011 / 0.009 |
| 588 KiB | 0.940 / 0.970 / 0.981 | 0.011 / 0.011 / 0.010 |

Serialization cost is set by the payload and not by where the bytes go,
which is what it should be. This was the control: had it moved, the
measurement would have been picking up CPU contention rather than the
network.

## One thing found after the fact, and labelled as such

P2 asked the wrong question. The ratio is noisy because it divides two noisy
numbers. The *absolute* cost of leaving the machine is much better behaved:

| payload | `win` − `self` |
|---|---|
| 48 KiB | +0.329 ms |
| 184 KiB | +0.522 ms |
| 588 KiB | +1.413 ms |

That is roughly **0.25 ms fixed plus about 2 µs per KiB** - a marginal rate
near 500 MB/s, which is a plausible figure for a virtual adapter. (The line
through the 48 and 588 KiB points is 2.01 µs/KiB with a 0.233 ms intercept,
which is 510 MB/s; least squares on all three points gives 2.05 µs/KiB and
499 MB/s. An earlier version of this sentence said 450 MB/s, which does not
follow from 2 µs/KiB - see the corrections below.) Payload does matter, but
for the crossing, not for the local stack.

**This was not pre-registered.** It is a description of the data, not a
tested hypothesis, and the next measurement of it should say so first.

## What this does and does not support

**Supported:**

* The process boundary costs about **1.3 ms** for a two-camera 224 px
  observation when the two sides are on different machines in the weakest
  sense available here.
* E5's loopback figure understates the real cost by roughly **0.5 ms**, not
  by an order of magnitude.
* The cost is in crossing the machine boundary. Routing through the network
  stack on one machine is free.

**Not supported:**

* **That 1.3 ms is cheap.** The protocol's own framing says the boundary is
  cheap relative to a VLA forward pass of tens of milliseconds, and nothing
  in PlugRL has ever measured one. That figure is borrowed, it is
  load-bearing, and E5's findings now say so in full. This experiment
  measured the boundary, not the ratio.

* **This is still not two machines.** The WSL2 guest and its host share a
  CPU, a memory bus and a hypervisor. `win` is the strongest rung available
  without a second computer and it is an intermediate point, not a LAN
  number. A physical NIC and a switch will cost more; how much more is
  unmeasured.
* Nothing about wide-area links.
* Nothing about many clients at once. Every run here is one client.
  E9 has since measured 1, 2, 4 and 8 clients against one server, but on a
  single machine, so it does not lift this limitation for the cross-machine
  numbers above - it only shows the server stays correct while several
  clients feed it.
* The server runs no policy, so these are transport-and-serialization
  numbers. A real run adds the forward pass, which makes the boundary look
  proportionally smaller - leaving it out is the conservative choice.

## What E7 found that it was not looking for

Every one of the 45 server logs ended the same way - this is
`results/srv_win_8971.log` lines 35-38, and `tail -1` gives the same last
line on all 45:

```
  FAIL  connection  (0 checks)
          ConnectionClosedError: no close frame received or sent
====================================================================
1 violation(s), 1 advisory note(s)
```

(Earlier versions of this document quoted those lines in the wrong order and
cut the clause `, 1 advisory note(s)`. That clause points at a second thing,
recorded below.)

The C++ client was dropping the socket without sending a WebSocket close
frame. It had gone unnoticed through every previous conformance run because
in all of them the *server* ran out of steps first and closed the connection
itself. E7 is the first experiment where the client finishes first - which
is also what a real env client does when it reaches its episode budget.

Fixing it surfaced two more things, in three different places:

* **The client** now sends a close frame with status 1000, per RFC 6455
  section 5.5.1.
* **The specification had a hole.** Section 7 described four ways the
  *server* closes and said nothing about the client stopping. It now has a
  section 7.5: a client may stop at any time, that is not an error, and it
  should say goodbye first.
* **The conformance server was stricter than the specification**, which is
  exactly the failure its two severity levels exist to prevent. It counted
  any client disconnect as a violation, including a clean one. A clean close
  is now recorded as satisfying 7.5, and a missing close frame as a note.

Verified one way; the other is retracted. Re-running the loopback cell
against the fixed client and the current conformance server - re-run while
correcting this document, not at measurement time - gives
`ok    7.5 a client that stops sends a close frame  (1 checks)` and the
summary line `no violations, 1 advisory note(s)` (the note is the `text` one
below). That check is reproducible from what is committed:

```bash
# in plugrl-protocol/examples, inside WSL2
python3 conformance_server.py --host 127.0.0.1 --port 8991 --steps 100000 --timeout 90 &
g++ -std=c++17 -O2 -o /tmp/e7_client plugrl_client.cpp
/tmp/e7_client 127.0.0.1 8991 30 1 224 2
```

but **no log of it is committed**: every log in `results/` predates the fix
and ends in the failure quoted above. The other half of the original
sentence, that the pre-fix binary reports `note 7.5` against the new
server, is **retracted**. That binary was not kept, no log of such a run exists
anywhere in the repositories, and the only support for the note path is the
source: `conformance_server.py:255-263` turns `ConnectionClosedError` into
an advisory rather than a violation. That is a code reading, not a
measurement, and the document should not have claimed otherwise.

None of this changes a single number above - the close frame is sent after
the last exchange is timed. It is recorded here because an experiment that
turns up a defect in the thing it is measuring with should say so.

### The advisory note in the same 45 logs: the two clients disagree about `text`

The clause this document used to drop points at a note that is also in every
one of the 45 logs, near the top (`results/srv_win_8971.log` lines 10-11):

```
  note  3.4 text is a numpy <U array, as plugrl-env-client sends  (2000 checks)
          infer: text arrived as a list; accepted, but a policy written against the array form would see something else
```

The C++ client packs the instruction strings as a msgpack array of strings
(`plugrl_client.cpp:674-676`: `p.str("text")`, `p.array(batch)`, then one
`p.str` per env); `plugrl-env-client` sends a numpy `<U` array. The
conformance server accepts both and records the difference as a note rather
than a violation, which is the right severity - nothing breaks, and the
severity levels exist for exactly this. But the two client implementations do
not put the same bytes on the wire for the same field, and a policy written
against the array form would see something else. A grep across every
Markdown file in the four repositories finds no mention of this, so it is
recorded here, where it first showed up.

It does not touch any number above: at batch 1 the field is one 12-byte
string beside 184 KiB of image arrays, and both forms unpack.

## Three deviations from the protocol: two deliberate, one not

1. **A third rung was added.** The protocol named `lo` and `win`. Measuring
   only those would have attributed the whole gap to the network, when in
   fact the network stack contributes nothing and the machine boundary
   contributes all of it. `self` is what separates them, and without it the
   headline sentence of this document would have been wrong.

2. **The quantitative claim rests on RTT, not on the throughput figure the
   protocol named.** Wall-clock throughput turned out to be dominated by
   process startup: at 20 exchanges, 0.36 s of wall clock covered 15 ms of
   actual exchanges. The run length was raised to 1000 exchanges and a
   startup-free `per_exchange_per_s` column added, but by then RTT with
   ranges over five repetitions was doing the work. Both columns are in
   `summary.tsv`; `exchanges_per_s` should not be compared across rungs.

3. **RTT was recorded as one mean column, not as mean/p50/p95. This one was
   not deliberate**, and it was not noticed until after publication. The
   protocol asks for RTT as "mean, p50, p95 over the samples after a
   3-exchange warm-up" and names the columns `rtt_mean_ms  rtt_p50_ms
   rtt_p95_ms`. The header actually in `results/summary.tsv` is:

   ```
   rung  payload_kib  rep  exchanges  pack_ms  rtt_ms  unpack_ms  wall_s  exchanges_per_s  per_exchange_per_s  valid
   ```

   One RTT column. `run.sh:106` takes field 7 of the client's TSV line, and
   the client emits `mean(pack_ms)`, `mean(rtt_ms)`, `mean(unpack_ms)` in
   fields 6-8 (`plugrl_client.cpp:820-822`); its p50 and p95 are printed in
   the human-readable block just above (`plugrl_client.cpp:811-819`) and then
   discarded - `run.sh` never writes the client's stdout to a file, so they
   are not recoverable from anything committed. **So every RTT figure in this
   document is a median over five repetitions of a per-run mean, and each
   `[min, max]` is the range of those five means, not of the samples.** A
   mean cannot show whether a spread is a tail or a shift, which is what p95
   was pre-registered for; the 588 KiB `win` row is where that would have
   mattered most, and it cannot be checked now without re-running. The
   warm-up half of the rule was honoured: `plugrl_client.cpp:725-726` starts
   recording at `step >= 3`, and a 30 exchange re-run reports 27 samples.

   `PROTOCOL.md` is pre-registered and must not be edited, so its record
   format block stands as written and is superseded by the header above.
   Recording the discrepancy here is the only correct move available.

## Five corrections to this document, made after publication

An adversarial read of these repositories turned up five defects here. All
five are corrected above and listed here rather than edited in silently.

1. **A marginal rate that did not follow from its own slope.** "About 2 µs
   per KiB" was paired with "near 450 MB/s"; 2 µs/KiB is 512 MB/s, and the
   fits on this data give 499-510 MB/s. The corrected figure is **near
   500 MB/s**. Arithmetic only - no conclusion moves.
2. **The quoted server log line was not verbatim.** It had the failure
   summary before the error, and it dropped `, 1 advisory note(s)`. Now
   quoted as the file has it.
3. **The dropped clause was hiding a second finding**, the `3.4` note that
   the C++ client and `plugrl-env-client` send `text` in different forms.
   Now recorded in its own subsection, which is where it should have been
   from the first version.
4. **Half of a verification claim had nothing behind it.** "The previous
   binary ... reports `note 7.5`" is retracted: no such binary and no such
   log exist. The `ok 7.5` half was re-run and holds, but no log of it is
   committed either, and the text now says so.
5. **A third protocol deviation was undisclosed** - RTT kept as one mean
   column instead of the pre-registered mean/p50/p95. It is now deviation 3,
   and it is the one deviation here that was not a deliberate choice.

Nothing in the measured numbers changed. Corrections 4 and 5 change what the
document claims to have established.

## Reproducing

```bash
bash run.sh 5 1000     # 3 rungs x 3 payloads x 5 reps, 1000 exchanges, ~10 min
bash run.sh 1 100      # a two-minute version
```

The run length has to be given. `run.sh` takes `[REPS] [EXCHANGES]` and
defaults to `5 120`, so a bare `bash run.sh` runs 120 exchanges per cell, not
the 1000 behind every row in `results/summary.tsv` - and 120 is close to the
short-run regime deviation 2 says was abandoned because process startup
dominated it. Earlier versions of this document showed the bare command with
the 1000-exchange description beside it.

`results/summary.tsv` has one row per repetition, including a `valid` column.
A run counts only if the client's own output confirms it completed the
requested number of exchanges - the trap E5's harness fell into when it
reported 118879 exchanges/s from a client that had died on connect. All 45
runs here were valid.
