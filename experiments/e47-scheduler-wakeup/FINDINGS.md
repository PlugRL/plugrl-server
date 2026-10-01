# E47: a scheduler that wakes on work answers an infer in 0.96 ms instead of 2.21, idles at 0.35% of a core, and trains the same weights

2026-10-01 · guangzhao · code: plugrl-server main 835f691 against
`fix/scheduler-wakeup` f2eb959 · protocol: [`PROTOCOL.md`](PROTOCOL.md)
(48566b5, after the pilot in `results/pilot.txt`, before the registered run)
· no amendment

---

## The result

E46's exploration found the server's scheduler making each infer wait up to
a millisecond. It slept `asyncio.sleep(0.0001)` between iterations. On Linux
asyncio waits in `epoll_wait`, whose timeout has millisecond resolution, and
Python rounds a positive timeout up to 1 ms. The fix gives the scheduler an
event:
- the server sets it whenever something the loop decides on changes: an
  infer queued, a feedback processed, a connection opened or closed, a stop
  requested;
- the loop waits on that event, with `asyncio.timeout(0.05)` as a backstop;
- it skips the wait when an infer is already ready.

Three servers, one client, HalfCheetah-sized observations. The figures are
medians of five runs:

| | round trip per infer | idle, of one core |
| --- | --- | --- |
| main (sleeps 0.1 ms) | 2.213 ms | 0.0595 |
| **fix** (waits on an event) | **0.964 ms** | **0.0035** |
| sleep0 (never sleeps; reference) | 0.891 ms | 0.9983 |

Every check and prediction holds:
- **V1:** every run completed, and both training runs' clients and servers
  exited 0.
- **P1:** fix is at most 1.2 ms and main at least 1.8.
- **P2:** fix is 0.073 ms from sleep0, within 0.2.
- **P3:** fix idles at 0.0035 of a core, under 0.02. sleep0 uses 0.998, which
  is why it is not the fix.
- **P4:** E43's cell (FPO · HalfCheetah, seed 0, 10 iterations) logged the
  same return at every iteration under main and under fix. The final weights
  are byte-identical. Their SHA-256, `6cbdcd21…`, is also E46's v1a's.
- **P5:** in training, the env client waited 1.312 ms per infer with fix and
  2.552 ms with main.

So collecting those 40,960 steps took **59.7 s instead of 111.9**, 1.87x
faster, for the same data.

---

## What the fix leaves

The remaining 0.07 ms between fix and sleep0 is the cost of waiting on an
event at all. A loop that never blocks does not pay it, and burns a core
instead.

The pilot measured a first version that waited with `asyncio.wait_for`. It
answered in 1.05 ms, because on Python 3.11 `wait_for` starts a task on
every call. The switch to `asyncio.timeout` was made after that number and
before the protocol, and the protocol says so.

Idle, main used 6% of a core. The 0.1 ms sleep, rounded up to 1 ms, made
about a thousand wake-ups a second. The fix wakes 20 times a second when
nothing happens.

---

## What this changes elsewhere

- **E5** (WSL2 Ubuntu, so also epoll) lowered the sleep from 1 ms to 0.1 ms.
  It measured 1.76x the throughput, with round trips between 0.53 and
  1.06 ms. A 0.1 ms sleep never waits 0.1 ms on Linux:
  - when the timer has already expired by the time the loop would block, it
    does not wait at all;
  - when it has not, it waits the full millisecond.

  Which one happens depends on the phase between the handler and the
  scheduler. That fits the spread, and it is where E5's gain came from. It
  was not waking on time.
- **E46.** In E46's Part B, v2's shorter wait came from this phase effect,
  not from reuse. That is now explained, as E46's findings said it would
  have to be.
- **Every plugrl-server run** on Linux with one fast client paid about
  1.2 ms per step, which on this cell was half the wall clock of collection.
  With slower environments or more clients the share is smaller, but the
  wait was there in every run.
- `RayAgentServer` still sleeps, because it is not given the event. Its
  loop is the same `RuntimeScheduler`, and passing it one is the same
  change.

---

## Files

- `PROTOCOL.md`: the pre-registration.
- `e47.sh`: all three parts, on guangzhao.
- `rtt_client.py`: the round-trip client.
- `compare_train.py`: the training comparison.
- `summarise.py`: judges every check and prediction.
- `results/`:
  - `rtt.tsv`, `idle.tsv`;
  - `train.txt` and `train/`: logs, TensorBoard events and env client
    timings, without the checkpoints;
  - `pilot.txt`.
