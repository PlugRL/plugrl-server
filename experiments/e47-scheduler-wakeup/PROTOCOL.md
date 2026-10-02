# E47 measurement protocol (pre-registered)

**Written 2026-10-01, after the pilot in `results/pilot.txt`, before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E46's exploration found that `WebSocketAgentServer`'s scheduler, which
sleeps `asyncio.sleep(0.0001)` between iterations, makes an infer wait up to
a millisecond. On Linux asyncio waits in `epoll_wait`, whose timeout has
millisecond resolution, and Python rounds a positive timeout up to 1 ms. A
server that never sleeps answered in 0.9 ms, and the shipped one in 2.2 ms.

The branch `fix/scheduler-wakeup` (f2eb959) makes the scheduler wait on an
event instead. The event is set whenever something it decides on changes,
and the wait is capped at 50 ms. **Does that remove the wait, without
burning a core and without changing what is trained?**

---

## Declared in advance: what was already known

1. **The arms**, all on guangzhao. Each lays a server source over the E43
   venvs with PYTHONPATH:
   - **main:** plugrl-server main, 835f691.
   - **fix:** f2eb959.
   - **sleep0:** main with `SCHEDULER_SLEEP_INTERVAL = 0.0` and nothing else.
     It is a reference for the fastest this loop can answer, not a candidate
     fix.

   The env client is guangzhao's 931ab56. `rtt_client.py` is E46's v1
   client.
2. **The pilot** (`results/pilot.txt`).
   - Round trips over 1,000 exchanges: main 2.07-2.32 ms, fix 0.98-1.00 ms.
     E46 measured sleep0 at 0.90 ms.
   - Idle CPU: main 0.057 of a core, fix 0.005.
   - Training, 2 iterations: the same returns and the same final weights,
     with SHA-256 matching E46's pilot. The env client waited 2.36 ms per
     infer with main and 1.19 ms with fix.
3. **The pilot shaped the fix.** Its first version used `asyncio.wait_for`,
   which starts a task per call on Python 3.11. It answered in 1.05 ms. The
   switch to `asyncio.timeout` was made after that number, and before this
   file.

---

## Design

`e47.sh`, on guangzhao, in this order:
- **Round trip:** `rtt 5 3000`. Five repetitions of 3,000 exchanges per arm.
  The arm order is main, fix, sleep0 on odd repetitions and the reverse on
  even ones. Each run gets a fresh server.
- **Idle:** `idle 5 20`. For each arm, the CPU a server with no client uses
  over 20 s, after 3 s of start-up. Same order.
- **Training:** `train 40960`. E43's cell (fpo-policy · FPO ·
  HalfCheetah-v5, E16's command lines, seed 0, 10 iterations), main and then
  fix, each alone. `compare_train.py` reads it.

---

## Checks

* **V1:** every run completes, and every training client and server exits
  0.

---

## Predictions, and what falsifies each

Medians are over the five repetitions.

**P1 - the wait is gone.** The median round trip is at most **1.2 ms** for
fix and at least **1.8 ms** for main.

> Grounds: known item 2. It fails if the fix leaves some of the 1 ms wait in
> place.

**P2 - nearly all of it.** fix's median is within **0.2 ms** of sleep0's.

> Grounds: known items 2 and 3 (0.98-1.00 against 0.90). What separates
> them is the cost of waiting on an event at all, not the epoll rounding.

**P3 - without burning a core.** fix's idle server uses at most **0.02** of a
core (median). sleep0's uses at least **0.5**, which is why it is not the
fix.

> Grounds: known item 2 for fix. For sleep0, from how it works: a loop that
> yields and is rescheduled at once never blocks.

**P4 - training is unchanged.** main and fix log the same return at all 10
iterations, and their final `model.safetensors` are byte-identical.

> Grounds: known item 2. With one client, when the loop wakes changes no
> data.

**P5 - and faster.** In training, the env client's wait per infer is at
most **1.4 ms** with fix and at least **1.8 ms** with main.

> Grounds: known item 2 (1.19 and 2.36).

**Reported, not predicted:** every round trip, the idle CPU figures for main,
the training's collect times and their ratio, and whether the final weights
match E46's v1a (SHA-256 `6cbdcd21…`).

---

## Reading order

V1, P1, P2, P3, P4, P5, then the reported figures.
