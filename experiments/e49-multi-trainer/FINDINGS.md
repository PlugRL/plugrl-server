# E49: RLinf, Stable-Baselines3 and CleanRL each train PlugRL env clients through plugrl-bridges, a C++ program with no third-party library included

2026-10-02 · guangzhao (RLinf on one RTX 5090; SB3 and CleanRL on CPU) ·
protocol: [`PROTOCOL.md`](PROTOCOL.md) (`3200e66`, after the pilots in
`results/pilot.txt`, before the registered runs) · no amendment

---

## The result

Three trainers that were not written for PlugRL, each through the
environment interface it already has, with 16 env clients and three seeds per
cell. The figures are each cell's gain by its status rule:

| trainer | interface | environment | gain per seed | bar |
| --- | --- | --- | --- | --- |
| RLinf PPO-MLP | `PlugRLEnv` | C++ Pendulum, 409,600 steps | +615, +763, +816 | +500 |
| SB3 PPO | `PlugRLVecEnv` | C++ Pendulum, 102,400 steps | +1,048, +1,071, +1,059 | +500 |
| SB3 PPO | `PlugRLVecEnv` | HalfCheetah, 409,600 steps | +1,433, +1,514, +1,551 | +200 |
| CleanRL PPO | `PlugRLVectorEnv` | HalfCheetah, 409,600 steps | +1,157, +1,341, +1,102 | +200 |

Every check and prediction holds (`results/registered/verdicts.txt`):
- **V1:** all 12 runs exited 0, with no Traceback, ClientLost or TimeoutError.
  Every slot logged its full count of episodes, each of the env's length.
- **V2:** every trainer's own record matches the bridge's episode log:
  - SB3's `monitor.csv`, episode for episode;
  - CleanRL's `charts/episodic_return`, episode for episode;
  - RLinf's `env/return`, epoch for epoch, 100 of 100.
- **V3:** in SB3 x Pendulum, each of the 16 C++ clients counted 6,400 steps
  and 32 episodes on its own. Its last-ten mean matches the log. (V3 does not
  apply under RLinf, as declared.)
- **P1-P4:** each cell learns on 3 of 3 seeds.

So three counts of the same episodes agree: the trainer's, the bridge's, and
the C++ clients'.

---

## RLinf on Pendulum is slow to start

E44's 102,400 steps would not have been enough. Over them, RLinf gained -14,
+12 and +63, reported here as the protocol requires. Every seed sits near
-1,150 for about 45 batches, then climbs to between -200 and -550 by batch
60. Afterwards it is unstable:
- seed 1 falls back to -1,388 at batch 69, then recovers;
- seed 2 sinks to around -900 for twenty batches.

SB3, at the same rl-zoo settings, has learned by batch 15. The pilot gave no
defect to chase:
- the bridge passed the same checks as in the other cells;
- RLinf trains HalfCheetah through it (E48).

RLinf's PPO-MLP differs from SB3's in its network, value clipping, initial
log-std and lack of gSDE. Why it learns Pendulum this way is a question about
RLinf's PPO at these settings, which E49 does not answer.

The 409,600-step budget was chosen after the pilot and declared before the
runs.

---

## What it cost

| cell | per run |
| --- | --- |
| RLinf x Pendulum, 100 epochs | 10.3, 9.9, 9.2 min |
| SB3 x Pendulum | 24-25 s |
| SB3 x HalfCheetah | 97 s |
| CleanRL x HalfCheetah | 77-78 s |

---

## What E49 does not show

* **Other trainers.** TorchRL, Tianshou, skrl and RLlib take a gymnasium
  VectorEnv too, but none was run. The claim stays at "an interface they
  accept".
* **Images, or more than one machine.** Every run here had its clients on
  guangzhao and sent states only. E48 put clients on another machine, under
  RLinf.
* **An independent comparison between the trainers.** Their algorithms and
  settings differ. The table says each learns, not which learns best.
* **RLinf ending its runs cleanly.** RLinf never closes its environments
  (E48), so the clients never hear stop and `run_bridged.sh` kills them.

Checkpoints and model files stay on guangzhao. Kept here, per run:
- `run.txt`, `trainer.log` and `episodes.csv`;
- the trainer's own record (`monitor.csv`, or TensorBoard events, with
  CleanRL's run directory shortened to `runs/tb` for Windows' path limit);
- the clients' logs.

Also kept:
- `pilot.txt`;
- `registered/verdicts.txt` (`summarise.py`, run on guangzhao);
- `bridges-src.sha256`.
