# E46: with reuse-feedback-obs each observation crosses once, so the size term of the crossing halves, and training is bit-identical

2026-10-01 · server on guangzhao, client on the E43 laptop, Tailscale direct
· code: plugrl-protocol#11, plugrl-server#109, plugrl-env-client#16 ·
protocol: [`PROTOCOL.md`](PROTOCOL.md) (`63f954b`, after the pilot in
`results/pilot.txt`, before the registered run) · no amendment

---

## The result

E43 measured the cost of crossing to another machine as a fixed latency plus
**twice** the observation's bytes over the link, because each observation
crossed twice: in the feedback that ends a chunk, and again in the next
infer. With `reuse-feedback-obs` (SPEC.md section 10.1), the infer leaves out
what the server already holds. Median `phys` round trip over five runs of
1,000 exchanges, at 23.7 MB/s:

| payload | v1 | v2 | v1's excess over states only ÷ v2's |
| --- | --- | --- | --- |
| states only | 2.91 ms | 3.08 ms | |
| 48 KiB | 7.34 | 5.41 | 1.79 (reported only) |
| 184 KiB | 18.53 | 11.58 | **1.81** |
| 588 KiB | 49.78 | 27.14 | **1.94** |

What it changes is the size term, and only that. On the same machine, and
in training, nothing measurable changes:
- **On one machine** (the `lo` and `ts` rungs) v2 is within 0.04 ms of v1
  up to 184 KiB. At 588 KiB it is 0.26 and 0.14 ms faster, because it does
  not pack, send and check a 588 KiB infer.
- **In training,** E43's cell (FPO · HalfCheetah, seed 0, 10 iterations)
  logged the same return at every iteration in all three runs: v1, v1 again
  and v2. The final weights are byte-identical (SHA-256 `6cbdcd21…`). v2's
  server served 40,920 of the observations from what it held, which is all
  but the 41 that followed a reset or began the run.

Every check and prediction holds:
- **V1:** 120 of 120 ladder runs are complete. The checker serving each one
  found no violations, which includes every one of v2's reuses.
- **V2:** every v2 infer frame was 269-374 bytes, and every v1 frame was the
  observation's size.
- **P1** holds: the ratio is between 1.6 and 2.4 at 184 and 588 KiB.
- **P2** holds: v2 saves 6.96 and 22.64 ms, 13% and 11% under P/B.
- **P3** holds: v1 is 2% and 8% under E43's s + 2P/B.
- **P4** holds: on `lo` and `ts`, within 0.25 ms, and at 588 KiB v2 is never
  slower.
- **V3, P5 and P6** hold.

---

## What crossing costs now

With the second copy gone, the model is a fixed latency plus **once** the
observation's bytes over the bandwidth. On this link, a 184 KiB camera
observation now costs 11.6 ms per step, against 18.5 ms with v1. That is
**12%** of the full-size pi0.5's 100 ms forward, where E43 had 21% on its
link (E10's ratio).

The measured saving is 11-13% short of P/B. E43's v1 was 3-10% short of
its model too: the throughput test is a bulk transfer, and an exchange moves
a little faster than it. The ratio in P1 does not involve B, and it is
within 0.2 of 2.

**This link is not E43's.** E43's path went across the campus LAN, and
today's goes to guangzhao's public address (`results/path.txt`). The
bandwidth was 23.7 MB/s against 18.0. v1 on this link against E43's
numbers:
- states only: 2.91 against 2.82 ms
- 48 KiB: 7.34 against 8.01
- 184 KiB: 18.53 against 21.38
- 588 KiB: 49.78 against 65.80

The fixed term is the same within 0.1 ms. The size term is smaller in
proportion to the bandwidth.

---

## Training is unchanged

The arms differed only in the env client: v1's sent every observation
(`--no-reuse-feedback-obs`), and v2's left out each row the server held.
The server offered the feature to all three. The env client decides per row
(plugrl-env-client#16). It leaves a row out only when that env's last
feedback on this connection did not end its episode, and the row is
byte-identical to what the feedback carried.

HalfCheetah truncates every 1,000 steps, so the run crossed 40 episode
boundaries. After each one the env client must send the reset observation,
because the feedback carried the terminal one. It did:
- The server read 40,961 infers: one per step, plus the one the env client
  sent before it saw the stop. Of those, 41 carried their observation in
  full: the first, and one after each of the 40 truncations. The other
  40,920 reused it. The pilot's 8,193 − 9 = 8,184 has the same structure.
- No server closed a connection for a resync, which a wrong reuse would have
  caused.
- The returns and weights are bit-identical, which a wrong row would have
  broken.

The unit test in plugrl-server#109 shows the same thing for PPO with two
envs on one connection. There, reused and reset rows share an infer, and a
reconstruction that swaps two rows fails on the weights.

---

## Found on the way: the server's scheduler sleeps up to a millisecond

This part was not pre-registered. In Part B, v2's env client waited
**1.43 ms** per infer and v1's **2.19**, so v2 collected 4,096 steps 30%
faster. Reuse cannot explain that: the observation is 17 floats.
`results/explore.txt` narrows it down with one controlled difference at a
time:
- **Not running three at once:** alone, v1 waits 1.78 and 1.79 ms and v2
  1.22 and 1.28.
- **Not the server's compute:** its own infer (0.70 against 0.67 ms) and
  feedback (23 against 20 µs) times agree. FPO's infer is no slower on a
  read-only decoded observation.
- **Not the garbage collector:** disabling it on either side leaves v1 at
  1.80-1.84 ms.
- **Not the bytes:** a client that sends `reuse` all false with the full
  observation is as fast as v2.
- **The scheduler's sleep:** with `SCHEDULER_SLEEP_INTERVAL` at 0 and
  nothing else changed, v1 and v2 both answer in **0.90-0.95 ms**. With the
  shipped 0.1 ms, v1 takes 2.18 ms and v2 1.60 ms.

The scheduler polls with `asyncio.sleep(0.0001)`. On Linux asyncio waits in
`epoll_wait`, whose timeout has millisecond resolution, and Python rounds a
positive timeout up to 1 ms. An infer that arrives while the 0.1 ms timer is
pending, with no further network event coming, waits up to a millisecond.
How often that happens depends on the phase between the connection handler
and the scheduler:
- v2's handler spends a few microseconds more before queueing the infer,
  which shifts that phase in its favour;
- a profiler slows everything and hides the gap entirely (v1 1.36 ms
  against v2 1.37 under cProfile).

So Part B's speed-up is not reuse's. The cost belongs to every plugrl-server
run: about 1.3 of a 2.2 ms round trip on one machine. E5 lowered the sleep
from 1 ms to 0.1 ms and measured a 1.76× throughput gain. That change did
not remove the wait, only made it phase-dependent, which may be why E5's
round trips were so noisy (0.53-1.06 ms). The ladder used the conformance
checker as its server, not plugrl-server, so Part A is unaffected.

A sleep of 0 busy-polls a core, so it is not the fix. The scheduler should
wake when there is work. That is a separate change, measured separately.

---

## Files

- `PROTOCOL.md`: the pre-registration.
- `bench_client.py`: E43's client plus `--reuse`.
- `ladder.sh`, `throughput.sh`, `part_a.sh`: Part A.
- `train_check.sh`, `compare_train.py`: Part B.
- `summarise.py`: judges every check and prediction from `results/`.
- `explore_*.sh`, `explore_*.py`: the exploration above, after the
  registered run.
- `results/`:
  - `ladder.tsv`, `throughput.tsv`, `path.txt`, `part_a.out`: Part A,
    including the checker's verdict for each run.
  - `train.txt` and `train/`: Part B's logs, TensorBoard events and env
    client timings, without the checkpoints. Their SHA-256 is in
    `train.txt`.
  - `explore.txt`: the exploration's numbers.
  - `pilot*`: the pilot.
