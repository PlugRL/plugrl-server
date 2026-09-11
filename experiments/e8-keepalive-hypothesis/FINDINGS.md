# E8: the learn step does not trip the keepalive, and the bug was real anyway

2026-09-11 · Windows 11, CPU only · FPO on HalfCheetah-v5 · plugrl-server and
plugrl-env-client at `main`

## In one sentence

A clean-machine run dropped its WebSocket connection, and the explanation
that suggested itself - a CPU-bound learn step holding the event loop past
the 20 s keepalive timeout - is **false**: learn steps of about 180 s produce
no timeout at all. The drop was a machine suspend. The silent data corruption
the drop exposed was real, and is the only part of the original diagnosis
that survived.

## The incident

Following the documented quickstart from fresh clones, the env client logged:

```
Connection closed during INFER/ACTION exchange. Error: sent 1011 (internal
error) keepalive ping timeout; no close frame received. Retrying...
```

Reading the code from there gives a tidy story. `websockets` defaults to a
20 s ping with a 20 s timeout. A learn step is CPU-bound. Therefore the
server misses its pong and kills a healthy connection.

Every step of that is plausible. The conclusion is wrong.

## What the measurement says

`run.sh` holds everything fixed and makes the learn step long. Gradient steps
per learn are `num_updates_per_batch * ceil(buffer_size / batch_size)`, so
raising the epoch count rather than the buffer keeps the fill short while
making the learn long - and it is the learn's *duration* the keepalive would
race against, not the buffer's size.

| run | buffer | updates/batch | gradient steps per learn | learns | learn duration | keepalive timeouts | reconnects |
|---|---|---|---|---|---|---|---|
| `short-learns` | 16384 | 16 (default) | 256 | 3 | roughly 5-24 s | **0** | **0** |
| `long-learns` | 4096 | 400 | 1600 | 5 | 177-190 s | **0** | **0** |

Two separate refutations, and the weaker run is enough on its own: one of
`short-learns`' three learn steps lasted about 24 s, **already past the 20 s
ping timeout**, and nothing closed. `long-learns` then put the learn step at
**nine times** the timeout, five cycles running, with the same result.

Durations are read off the gaps between the client's periodic timing lines,
which are emitted every 30 s; a gap of 53.7 s contains one 30 s interval plus
about 24 s of waiting. That makes them approximate, and approximate is
sufficient to separate 24 s from 20 s in the direction that matters, because
the hypothesis predicts a close and there was none.

The reason is one line in `server/training_backend.py`:

```python
step, log_dict = await asyncio.to_thread(self._algorithm.learn)
```

**Learning already runs off the event loop.** The loop stays free to answer
pings for the whole learn. The hypothesis was not merely unproven, it was
contradicted by code that was there to read.

## What actually caused the drop

The client's own log, at the two lines either side of the failure:

```
2026-09-11 12:28:16.237 | INFO | Intermediate rollout timing summary: env_steps=5353 ...
2026-09-11 14:21:53.007 | WARNING | Connection closed during INFER/ACTION exchange ...
```

Those timing lines are emitted every 30 s, without a break, from 12:24:14 to
12:28:16. Then **1 hour 53 minutes of nothing**, and the next line is the
failure. A process that was merely slow would still have logged. This one was
frozen: the machine suspended with the run open, and when it came back both
ends had long since stopped hearing from each other.

Two further details the original diagnosis had backwards:

* the traceback is in `websockets/sync/connection.py`, in the **client**
  library. `sent 1011` means the *client* closed the connection because the
  *server* had not answered the client's pings. The server's ping settings
  are not what failed.
* the server log for the same run contains no keepalive line at all.

So a fix that turned the server's keepalive off would not have prevented this
incident. It was written, and is not in the change that shipped.

## What survived, and is worth more than the hypothesis was

The drop was real, and what it exposed does not depend on why it happened.

`prev_node_map`, `step_state_map`, `terminated_map`, `truncated_map` and
`last_obs_map` are local to the connection handler. A reconnect gets a new
handler and five empty maps. The env client, meanwhile, retried its held
`feedback` on the new connection after every close except an explicit resync -
so the server completed that transition from an empty observation and stored
it. No exception, no warning, one corrupt transition per reconnect.

That is a real defect, reachable by any cause of reconnection: a suspend, a
flaky link, a server restart. It is fixed in the client, which now drops a
held feedback rather than resending it, and the server now says so when a
feedback arrives with no step state. The protocol gained section 7.6, which
states the rule in terms of the reconnect rather than any particular cause.

**E6 was checked for contamination and is clean:** one connection per seed,
zero reconnects across all three.

## What this does and does not support

**Supported:**

* A learn step, however long, does not close a connection. Measured to 190 s,
  nine times the timeout.
* The reconnect state-loss path is real; the code is explicit about it, and
  the client's retry made it reachable.

**Not supported:**

* Anything about what *does* trip the keepalive in normal operation. One
  incident, and its cause was a suspended machine, which is not a
  steady-state condition worth designing against.
* Any claim about frequency. This happened once, on a machine that sleeps.
* Anything on hardware other than this one. A different policy on a different
  machine may block the loop somewhere this one does not - but it will not do
  it inside `learn`, which is the specific claim tested here.

## The methodological note

The first diagnosis was reasoned from code to a conclusion that fit the
symptom, and it fit well enough that the fix, its test, its specification
clause and four pull requests were all written before anything measured it.
The measurement took twenty minutes and reversed it.

What makes this recoverable rather than embarrassing is that the *consequence*
was verified independently of the cause: the state-loss path was read out of
the code and reproduced in a unit test, not inferred from the incident. The
part that rested on the incident alone is the part that was wrong.

## Reproducing

```bash
bash run.sh short-learns 16384 49152 1 8617 16    # default epochs, ~9 min
bash run.sh long-learns   4096 20480 1 8615 400   # long learns, ~15 min
```

The last argument is `num_updates_per_batch`. `results/` holds both runs and
the original incident log. A run counts only if the client exits zero, and
the script reports keepalive timeouts, 1011 closes, client reconnects and
server-side missing-step-state warnings.
