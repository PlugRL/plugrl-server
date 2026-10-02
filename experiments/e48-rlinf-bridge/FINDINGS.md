# E48: RLinf, through the bridge, trains environments that PlugRL env clients run outside its cluster - on its own machine and on a laptop with no torch, Ray or RLinf

2026-10-01/02 · RLinf on guangzhao (one RTX 5090); env clients on guangzhao
(local) or on the E43 laptop over Tailscale (cross) · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`c64ad7c`, after the pilot in
`results/pilot.txt`, before the registered runs) · amendment:
[`AMENDMENT.md`](AMENDMENT.md) (an orchestration fault in the cross arm; no
judged figure is affected)

---

## The result

RLinf's PPO-MLP, through plugrl-rlinf's bridge (fca69db), trained
HalfCheetah-v5 run by 16 plugrl-env-client processes: 100 epochs of 4,096
frames, 409,600 steps. Mean `env/return` of the first episode batch against
the mean of the last five:

| seed | local | cross |
| --- | --- | --- |
| 0 | -284 -> 449 (**+733**) | -274 -> 450 (**+724**) |
| 1 | -290 -> 485 (**+775**) | -274 -> 463 (**+737**) |
| 2 | -268 -> 481 (**+749**) | -244 -> 496 (**+740**) |

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** each run's log has 16 `plugrl-rlinf: slots` lines. They came from
  127.0.0.1 in the local arm, and from the laptop's Tailscale address,
  100.65.224.34, in the cross arm.
- **V2:** every run logged 100 epochs, with no Traceback, ClientLost or
  TimeoutError.
- **P1:** the local arm learns on 3 of 3 seeds, against a bar of +200.
- **P2:** the cross arm learns on 3 of 3 seeds.

The laptop's side was a 276 MB venv with mujoco and plugrl-env-client, and no
torch, Ray or RLinf. RLinf ran on guangzhao as it always does. Its
environment worker reached the 16 environments only through the PlugRL
protocol.

---

## What it cost

| | local | cross |
| --- | --- | --- |
| median s per epoch | 3.71, 3.69, 3.69 | 23.68, 24.62, 22.42 |
| run, start to RLinf's exit | 6.5, 6.4, 6.4 min | 41.1, 50.3, 55.9 min |
| Tailscale ping before / after | | 46/59, 44/48, 50/42 ms |

Crossing made an epoch 6.4x slower. The step itself is unchanged; the cost is
the exchange. Each epoch is 256 lockstep steps, and each step waits for all 16
clients across a 42-59 ms link. That is about 90 ms per step, against 14 ms
locally. The run times spread more than the medians do: some epochs were much
slower than the median. This is the same shared Wi-Fi link that E43 found
does not hold a constant cost.

The bridge is lockstep. A step finishes when its slowest client has fed back.
That is why the crossing costs more here than in E43, where plugrl-server
answers each client as soon as it can.

---

## What it means

The bridge does what it was built to do. A trainer whose environments must
normally run inside its own Ray cluster, in its own Python, trained
environments that ran on a machine that has none of that. The learning curves
of the two arms overlap seed for seed. Where the environment is stepped
changes the time a step takes, not what it produces.

---

## What the run found, beyond its predictions

1. **RLinf never closes its environments when a run ends**
   (`rlinf/workers/env/env_worker.py` calls no `close()`). In its own cluster
   that costs nothing. Clients outside it see the connection drop with no
   close frame, never get `plugrl-server-stop`, and reconnect for good, as
   SPEC.md tells them to: 3,190 attempts in 25 minutes after cross seed 0
   ended. The local arm never showed this, because `run_guangzhao.sh` kills
   its clients when RLinf exits.
2. **Git Bash rewrites POSIX paths passed to Windows programs.** The cross
   arm's poll looked for `D:/Program Files/Git/home/...` on guangzhao and
   never saw a run end. `AMENDMENT.md` records how the watcher stood in for
   it. plugrl-bridges' `cross_clients.py` now refuses such a path.

---

## What E48 does not show

* **Any trainer but RLinf, or any environment but HalfCheetah.** E49 runs
  SB3 and CleanRL, and the C++ Pendulum client.
* **Images through the bridge.** The observations are 17 floats.
* **Identical trajectories across machines.** The two machines' MuJoCo
  builds differ (E43), so the arms are compared by the status rule.
* **A remote client that survives the run's end.** See item 1 above. The fix
  belongs in the bridge, as a stop on exit, or in the client, as a bounded
  wait for a server.

Checkpoints stay on guangzhao, in `~/zuogou/plugrl/e48/<run>/rlinf-logs/`.
Kept here:
- each run's `run.txt`, `rlinf.log` and TensorBoard events;
- the clients' logs;
- `e48.txt` and `run_e48.out`;
- `stop-watcher.sh` and `stop-watcher.txt`;
- `pilot.txt` and `verdicts.txt` (`summarise.py`).
