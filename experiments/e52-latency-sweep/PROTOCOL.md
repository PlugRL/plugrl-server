# E52 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilot in `results/pilot.txt`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

The paper claims that latency between environment and trainer slows a run
without changing what it collects. That rests on a design argument: the env
client waits for every action. E43 adds indirect evidence: two arms on
different machines learned alike, but their physics differ in the last bit,
so their data could not be compared. So:

**On one machine, with latency added between the env client and the bridge,
does training end on the same weights? And does the time grow as the
protocol says it should, by two one-way delays per step?**

---

## Declared in advance: what was already known

1. **The code.**
   - `train.py` is E50's, with one added option: `--client-port`, the port
     the env client connects to.
   - The client is E50's `gym_client.py`. The bridge is plugrl-bridges as
     synced to guangzhao for E49 (`79ec044`).
   - SB3 2.9.0 PPO with rl-zoo's Pendulum-v1 settings, 16 envs, 102,400
     steps, one torch thread, CPU.
2. **`delay_proxy.py`** forwards each chunk D ms after it arrived, in each
   direction. It does not limit bandwidth. It sleeps with `asyncio.sleep`,
   which on Linux waits in `epoll_wait` and rounds up to the millisecond
   (E47). A step is one action out and one feedback-and-infer back, so it
   should take 2D longer.
3. **E50, on this machine.** The in-process arm ends on these weights:
   - seed 0: `ec5b8226...`;
   - seed 1: `062e92e3...`;
   - seed 2: `3a8d968f...`.

   E50's bridge arm ends on the same weights.
4. **The pilot, seed 9** (`results/pilot.txt`).
   - Direct and 5 ms both ended on `3721266c...`, E50's seed-9 hash on
     guangzhao.
   - Direct took 16.1 s and 5 ms took 92.1 s. That is 11.87 ms more per
     vector step (6,400 of them), against 2D = 10 ms.

---

## Design

On guangzhao, through `run.sh`, one run at a time:
- seeds 0, 1 and 2;
- for each, the arms direct (no proxy, as E50), then 0, 1, 5 and 25 ms one
  way.

That is 15 runs.

---

## Checks

* **V1 - complete.** Every run has `steps` 102,400 and `client_rc` 0, and its
  log has one `plugrl-bridges: slots 0-15` line.

---

## Predictions, and what falsifies each

**P1 - latency leaves the weights alone.** All 15 runs end on E50's
in-process weights for their seed.

> Grounds: known items 3 and 4. The client blocks on every action, so the
> data do not depend on when an action arrives. The bridge is lockstep, so
> neither does the order of the data.

**P2 - and costs two one-way delays per step.** For D = 1, 5 and 25 ms,
the time per vector step beyond the direct run's lies between 2D and
2D + 3 ms, on every seed.

> Grounds: known items 2 and 4. The pilot's excess over 2D was 1.87 ms at
> 5 ms: the proxy's own forwarding, plus the sleep rounding up on each leg.

**Reported, not predicted:**
- the 0 ms arm's time beyond the direct run's, the proxy's own cost;
- each arm's excess over 2D.

---

## Reading order

V1, P1, P2, then the reported figures.
