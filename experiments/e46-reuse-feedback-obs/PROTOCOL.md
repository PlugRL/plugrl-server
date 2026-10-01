# E46 measurement protocol (pre-registered)

**Written 2026-10-01, after the pilot in `results/pilot.txt`, before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The questions

E43 found that crossing to another machine costs a fixed latency plus
**twice** the observation's bytes over the link, because every observation
crossed twice: once in the feedback that ends a chunk, once more in the next
infer. plugrl-protocol#11 adds `reuse-feedback-obs` (SPEC.md section 10.1),
with which an infer leaves out the observations the server holds.
plugrl-server#109 offers it, and plugrl-env-client#16 uses it. So:

**A. With the feature, is the size-proportional part of the crossing cost
halved?**

**B. Does using it change what is trained?** It should change the bytes on
the wire and nothing else.

---

## Declared in advance: what was already known

1. **The code.**
   - plugrl-protocol#11 at 4c877bc: the conformance checker, which serves the
     ladder, with `--features reuse-feedback-obs`.
   - plugrl-server#109 at 75be7a8, which offers the feature and counts reused
     rows in `server/reused_observations`.
   - plugrl-env-client#16 at bb890c7, which uses the feature unless given
     `--no-reuse-feedback-obs`.
   - On guangzhao, all three are laid over the venvs E43 used, with
     PYTHONPATH (`train_check.sh`). The versions are mujoco 3.14.0,
     gymnasium 1.3.0, torch 2.7.1+cpu and numpy 2.3.2.
   - `bench_client.py` is E43's bench client plus `--reuse`. Without the flag
     it sends what E43's sent.
2. **The machines and the link** (`results/pilot.txt`).
   - The server is guangzhao, as in E43.
   - The client is the same Windows 11 laptop, on Cornell's network at
     10.48.165.215.
   - Tailscale connects them directly, but **not by E43's path**. E43's went
     through the campus LAN. Today's goes to guangzhao's public address,
     128.84.216.41:41641, after starting on a DERP relay through New York.
     `tailscale ping` gives 8 ms.
   - So the `phys` rung is not E43's link, and its numbers are compared with
     E43's only as reported figures.
3. **Pilot throughput**, laptop to guangzhao, three runs of 16 MiB: 20.2, 19.1
   and 19.4 MB/s.
4. **The pilot ladder.** One repetition of 100 exchanges, median round trip in
   ms, v1 then v2:

   | payload | `lo` | `ts` | `phys` |
   | --- | --- | --- | --- |
   | states only | 0.17, 0.25 | 0.14, 0.17 | 4.08, 2.78 |
   | 48 KiB | 0.16, 0.21 | 0.16, 0.23 | 7.02, 5.87 |
   | 184 KiB | 0.30, 0.44 | 0.32, 0.32 | 20.36, 11.70 |
   | 588 KiB | 0.66, 0.61 | 1.18, 0.75 | 59.85, 31.24 |

   - v2's infer frame is 269-374 bytes at every payload. v1's is the
     observation's size.
   - The `phys` states-only pair, 4.08 against 2.78 ms, sends the same bytes
     within 92. So the pilot's run-to-run noise at `phys` is about a
     millisecond, which is why 48 KiB carries no prediction below.
   - At 184 and 588 KiB, v1's excess over states only (their mean, 3.43 ms)
     is 2.05 and 2.03 times v2's.
5. **The pilot training** (`train_check.sh 8192`, two iterations, seed 0).
   - v1a, v1b and v2 logged the same return at both iterations, -292.856842
     and -278.121399.
   - Their final `model.safetensors` have one SHA-256.
   - v2's server counted 8,184 reused observations out of 8,192 frames. v1a's
     and v1b's counted 0.
   - None of the three servers closed a connection for a resync.
6. **Determinism on guangzhao.** E43 ran seed 0 of this cell three times and
   got the same return at every iteration. Pilot item 5 agrees.

---

## Design

**Part A first**, so that training does not share guangzhao with the lo and
ts rungs:

- **Throughput.** Five transfers of 64 MiB, laptop to guangzhao
  (`throughput.sh 5 64`).
- **The ladder.** `ladder.sh 5 1000`:
  - three rungs, as in E43: `lo` and `ts` from guangzhao itself, `phys` from
    the laptop
  - four payloads: states only, 48, 184 and 588 KiB
  - two arms in every cell: v1 (`bench_client.py`) and v2
    (`bench_client.py --reuse`). The server offers the feature to both, so
    the arms differ only in the client.
  - five repetitions of 1,000 exchanges. The rung order and the arm order
    both reverse on even repetitions.

**Then Part B**, on guangzhao alone: `train_check.sh 40960`. This is E43's
cell, `fpo-policy` · FPO · HalfCheetah-v5 with E16's command lines, seed 0,
for 10 iterations of 4,096 steps. Three runs go at once:
- **v1a:** the env client sends every observation;
- **v1b:** the same again, as the control for determinism;
- **v2:** the env client reuses the observations the server holds.

`compare_train.py` reads Part B, and `summarise.py` reads both parts.

---

## Checks

* **V1 - a ladder run counts** only if its client completed its 1,000
  exchanges and exited 0. The checker serving it must also have ended with
  no violations.
* **V2 - the arms are what they say.** Every v2 run's infer frame is under
  1 KiB. Every v1 run's is at least the observation's size.
* **V3 - Part B's runs are complete.** Each of the three logged 10
  iterations, each client exited 0, and no server closed a connection for a
  resync.

---

## Predictions, and what falsifies each

Write s for the median `phys` round trip with states only, pooled over both
arms' ten runs, and B for the median of the five throughputs. For a payload
of P bytes, P is the feedback frame's size in that cell.

**P1 - the factor of two is gone.** At 184 and 588 KiB, the ratio
(median v1 `phys` − s) / (median v2 `phys` − s) is between **1.6 and 2.4**.

> Grounds: the mechanism, and known item 4 (2.05 and 2.03). This test does
> not depend on B. It fails if the second copy was not what made the term
> twice as large, or if leaving it out costs something comparable.

**P2 - v2 saves one observation's bytes over the link.** At 184 and 588 KiB,
median v1 `phys` − median v2 `phys` is within **±25%** of P / B.

> Grounds: E43's model, and known items 3 and 4, where it was 11% and 7%
> under.

**P3 - v1 still fits E43's model.** At 184 and 588 KiB, median v1 `phys` is
within **±25%** of s + 2P / B.

> Grounds: E43's P6, which held 3-10% under. This prediction is on a
> different path (known item 2), so it is a replication on a new link and
> not a repeat.

**P4 - on one machine the feature neither helps nor hurts measurably.**
- At states only, 48 and 184 KiB: on `lo` and on `ts`, median v2 is within
  **0.25 ms** of median v1.
- At 588 KiB: median v2 is at most 0.25 ms **above** median v1.

> Grounds: known item 4. On one machine v2 skips packing, sending and
> validating one observation, and adds a merge on the server, so at 588 KiB
> it may come out ahead by more than 0.25 ms. The pilot's `ts` pair at
> 588 KiB, 1.18 against 0.75 ms, is the case. That is why the bound there is
> one-sided.

**P5 - training is unchanged.** v1a and v1b log the same return at all 10
iterations, and their final `model.safetensors` are byte-identical (the
control). v2 then logs v1a's return at all 10 iterations, and its final
`model.safetensors` is byte-identical to v1a's.

> Grounds: known items 5 and 6. If the control fails, the run says nothing
> about reuse. That is recorded as an inconclusive Part B, not as a failure
> of P5.

**P6 - the feature was used.** At its last log, v2's
`server/reused_observations` is at least **98%** of the frames, and v1a's and
v1b's are **0**.

> Grounds: known item 5, at 99.9%. One infer per step (replan 1). An infer
> after a reset must send its observation, and HalfCheetah truncates every
> 1,000 steps, so about 41 of 40,960 are sent.

**Reported, not predicted:**
- every cell's pack, round-trip and unpack times, and its frame sizes
- the 48 KiB row
- the throughputs
- the `phys` figures beside E43's
- E10's ratio, recomputed with v2's `phys` at 184 KiB against the full-size
  pi0.5's 100 ms forward
- Part B's returns and wall clock

---

## Declared deviations allowed in advance

1. If the laptop's Tailscale path changes during the ladder, for example
   back to a DERP relay, the repetition in progress is rerun once. This is
   recorded in `AMENDMENT.md`.
2. Part B may be rerun once if a process dies for a reason outside the
   experiment. This is also recorded in `AMENDMENT.md`.

---

## Reading order

V1, V2, P1, P2, P3, P4; then V3, P5, P6; then the reported figures.
