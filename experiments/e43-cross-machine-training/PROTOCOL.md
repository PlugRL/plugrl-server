# E43 measurement protocol (pre-registered)

**Written 2026-09-28, after the pilot in `results/pilot.txt`, before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The questions

PlugRL's home page says the env clients can run on another machine. Every
cell of the coverage figure was trained with both sides on one machine, and
E7's crossing was a VM to its own host. So:

**A. With the env clients on another physical machine, does training still
learn?**

**B. What does crossing to that machine cost per exchange, and what is the
cost made of?**

---

## Declared in advance: what was already known

1. **The cell.** `fpo-policy` · FPO · HalfCheetah-v5 from the coverage
   figure, with E16 Phase A's command lines (`server_side.sh`): 409,600
   steps, buffer 4,096, so 100 iterations. One server per seed, and the same
   seed for the server and the client. Its status rule, as the figure has
   it: it **learns** if the mean return over iterations 91-100 is at least
   **+200** over iteration 1, on at least **2 of 3** seeds.
2. **The machines** (`results/pilot.txt`).
   - The server runs on guangzhao, a wired Linux workstation.
   - The second machine is a Windows 11 laptop on Wi-Fi 6. Its env client
     runs natively on Windows, not in WSL.
   - Both are on the same campus network and joined by Tailscale
     peer-to-peer, not relayed. guangzhao's public address refuses inbound
     connections, so Tailscale is the path.
   - The env client is at the same commit on both machines, with the same
     mujoco, gymnasium and numpy versions.
3. **The pilot's training.** Over 8,192 steps both arms ran, and the clients
   exited 0.
   - Per step on the server's clock: 2.94 ms local, 5.70 ms cross.
   - Per call on the client's side, local then cross:
     - infer wait: 2.5 ms, 4.6 ms
     - env step: 0.09 ms, 0.55 ms (the laptop steps MuJoCo slower)
     - feedback send: 0.07 ms, 0.45 ms
4. **Determinism.** On guangzhao, seed 0 run three times gave the same return
   at every iteration. Across the two machines the returns differ from the
   first iteration. The reason is the physics, not the transport:
   - With no network involved, the same seed and actions give observations
     that differ by 8.9e-16 after one step, 4e-7 by step 50 and order 1 by
     step 100.
   - This is the last bit of the floating-point math on two platforms,
     amplified by a chaotic system.
   - So the arms are compared by the status rule, not trajectory by
     trajectory.
5. **The pilot's ladder** (one repetition, 200 exchanges).
   - `lo` and `ts` are within 0.25 ms of each other at every payload.
   - `phys` is 3.8 ms with states only, then 9.1, 21.8 and 55.7 ms at 48, 184
     and 588 KiB.
   - Pilot throughput was 19.0, 21.7 and 23.5 MB/s. With it, `phys` fits a
     fixed cost plus twice the observation's bytes over the link's
     throughput. The observation crosses twice per exchange: once in the
     previous feedback, once in the infer.
   - **This model was suggested by the pilot, and it is declared as such**.
     P6 tests whether it holds on fresh measurements.

---

## Design

**Part B first**, so that the training does not share the link with it:

- **Throughput.** Five transfers of 64 MiB, laptop to guangzhao
  (`throughput.sh`).
- **The ladder.** `ladder.sh`, with plugrl-protocol's conformance server on
  guangzhao and `bench_client.py` on either machine:
  - three rungs: `lo` and `ts` from guangzhao itself, `phys` from the laptop
  - four payloads: states only, 48, 184 and 588 KiB
  - five repetitions of 1,000 exchanges each; the rung order reverses on
    even repetitions

**Then Part A.** The local arm and the cross arm run at the same time, seeds
0-2, with 409,600 steps each (`server_side.sh local`, `server_side.sh cross`,
`client_side.sh`):

- **The local arm**: servers and clients on guangzhao.
- **The cross arm**: servers on guangzhao, clients on the laptop.

`summarise.py` reads everything.

---

## Checks

* **V1 - the arms are what they say.**
  - Every cross client connected to guangzhao's Tailscale address, and every
    local client to 127.0.0.1.
  - The six servers logged the same algorithm configuration apart from the
    port and the name.
* **V2 - a ladder run counts** only if its client completed its 1,000
  exchanges and exited 0.

---

## Predictions, and what falsifies each

**P1 - both arms run end to end**: on all six seeds, 100 iterations logged,
five checkpoints, no traceback, and the client exited 0.

**P2 - the cross arm learns**, by the status rule. **P3 - the local arm
learns**, by the same rule.

> Grounds: known item 1. E16's control learned this cell on one machine, from
> about -300 to 1,346-2,050. Where the environment is stepped changes the
> timing, not the data a step produces; item 4's differences are
> floating-point, not systematic. Each is falsified if fewer than two seeds
> clear the bar.

**P4 - the cross arm's extra time is the pilot's per-step cost.** For each
seed, the cross arm's span from iteration 1 to iteration 100 on the server's
clock minus the local arm's, against 99 x 4,096 x 2.76 ms = **1,119 s**.

> Grounds: known item 3. This fails if crossing costs something that grows
> over a run and that two iterations do not show, such as queueing or
> backlog. Holds if the difference is within **±25%** (839-1,399 s) on at
> least 2 of 3 seeds.

**P5 - `lo` and `ts` agree.** At every payload, the medians of the five
repetitions are within **0.25 ms**.

> Grounds: known item 5, and E7's `self` rung. On one machine a Tailscale
> address is short-circuited like any other local one.

**P6 - the crossing cost is fixed latency plus bytes over bandwidth.** Take
B = the median throughput of the five transfers. For 48, 184 and 588 KiB,
the median `phys` round trip is predicted as the median `phys` round trip
with states only, plus 2 x (the observation's image bytes) / B. It holds if
the measured median is within **±25%** of the prediction at all three
payloads.

> Grounds: known item 5, and suggested by it (declared above).

**Reported, not predicted:**
- every rung's pack, round-trip and unpack times
- the throughputs
- the env clients' timing breakdown in both arms
- the returns at iterations 1, 50 and 91-100 for every seed
- wall clock
- E10's ratio recomputed with `phys` at 184 KiB against the full-size
  pi0.5's 100 ms forward

---

## Declared deviations allowed in advance

1. One restart of any cross-arm seed that dies for a reason outside the
   experiment, such as the laptop's Wi-Fi dropping or the laptop sleeping.
   It is recorded in `AMENDMENT.md`. The local arm's matching seed is rerun
   with it, so that the two arms stay concurrent.

---

## Reading order

P1, V1, the status rule, P2, P3, P4; then V2, P5, P6; then the reported
figures.
