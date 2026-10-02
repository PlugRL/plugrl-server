# E52: latency between env client and bridge changes the time, not the weights - up to 25 ms each way, SB3 ends on its in-process weights

2026-10-02 · guangzhao (bridges-venv: torch 2.14.1+cpu, SB3 2.9.0,
gymnasium 1.3.0; plugrl-bridges 79ec044) · protocol: [`PROTOCOL.md`](PROTOCOL.md)
(`ce807c4`, after the pilot in `results/pilot.txt`, before the registered
runs) · no amendment

---

## The result

SB3's PPO trained 16 Pendulum environments through the bridge, with the env
client connecting either directly or through `delay_proxy.py`. The proxy
holds every byte for D ms in each direction.

| seed | weights, every arm | direct | 0 ms | 1 ms | 5 ms | 25 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `ec5b8226...` | 16.3 s | 16.8 s | 35.7 s | 92.1 s | 355.6 s |
| 1 | `062e92e3...` | 16.1 s | 17.2 s | 35.4 s | 92.7 s | 355.9 s |
| 2 | `3a8d968f...` | 16.2 s | 16.9 s | 35.4 s | 92.4 s | 356.1 s |

Per vector step (6,400 a run), beyond the direct run, and how far that is
above 2D (two one-way delays):

| D | per step | beyond 2D |
| --- | --- | --- |
| 0 ms | +0.07 to +0.17 ms | +0.07 to +0.17 |
| 1 ms | +3.00 to +3.04 ms | +1.00 to +1.04 |
| 5 ms | +11.85 to +11.96 ms | +1.85 to +1.96 |
| 25 ms | +53.02 to +53.10 ms | **+3.02 to +3.10** |

Every check and prediction holds except one:
- **V1:** all 15 runs are complete, and each bridge run served one
  connection.
- **P1 holds.** Every run, at every delay, ends on E50's in-process weights
  for its seed. A run with 25 ms each way between environment and trainer
  takes 22 times as long and ends on the same bytes as a run with the
  environments in SB3's own process.
- **P2 is falsified, at 25 ms only.** The band was 2D to 2D + 3 ms. At 1 and
  5 ms every seed is inside it. At 25 ms all three seeds are above it by
  0.02-0.10 ms.

---

## Where the excess comes from

The excess over 2D grows with D, from 0.1 ms to 3.1 ms. The pilot gave only
one point, 1.87 ms at 5 ms, and P2's band assumed the excess was a constant.
It is not.

After the runs, `proxy_rtt.py` measured the proxy alone. It sent 4 KiB
messages through `delay_proxy.py` to a TCP echo server, 300 round trips per
delay (`results/proxy_rtt.txt`). Beyond 2D, the median was:

| D | one message | two messages 0.2 ms apart |
| --- | --- | --- |
| 0 ms | 0.03 ms | 0.28 ms |
| 1 ms | 0.48 ms | 0.77 ms |
| 5 ms | 1.00 ms | 1.04 ms |
| 25 ms | 0.96 ms | 1.29 ms |

So the proxy itself accounts for about 1 ms of each step's excess. That is
its two `asyncio.sleep` calls, each rounded up in `epoll_wait`, as E47 found
for the server's scheduler. The rest grows with D:
- 0.2-0.5 ms at 1 ms;
- about 0.9 ms at 5 ms;
- about 1.8 ms at 25 ms.

It is **not explained** here. It lies between the proxy and the trainer: in
the WebSocket client, the bridge's event loop, or the order in which they
wake. A tighter claim would need a delay that does not sleep in Python, such
as `tc netem`, which needs root on guangzhao.

---

## What it means

- **The claim holds as stated: latency changes time, not data.** The client
  blocks on every action, and the bridge is lockstep. So when an action
  arrives decides when a step happens, never what it produces.
- **Its price is two one-way delays per step, plus a little.** On a 25 ms
  link that is 53 ms a step, against 2.5 ms of computation. A lockstep
  trainer on a long link spends nearly all its time waiting. That is E48's
  6.4x slowdown across Wi-Fi, and the reason an asynchronous server (E43:
  2x) or a deeper pipeline is the design for distant environments.
- **Measure latency without sleeping where you can.** The tool that adds
  latency adds its own (E47).

---

## What E52 does not show

* **Real links.** The proxy adds latency but not bandwidth limits, jitter
  or loss. E43 and E48 measured a real Wi-Fi link.
* **An environment that keeps moving.** The client waits for each action.
  A robot does not, and latency then changes the data. That is out of
  scope (the paper's limitations).
* **The excess's cause beyond the proxy's own share.** See above.

Kept here:
- each run's `result.json`, `monitor.csv`, `episodes.csv`, logs and proxy
  logs;
- `registered.out`, `load.txt`, `verdicts.txt` (`summarise.py`);
- `proxy_rtt.txt` (`proxy_rtt.py`, run after the registered runs).
