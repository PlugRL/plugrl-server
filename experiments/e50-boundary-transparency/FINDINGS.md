# E50: with its environments behind the bridge, SB3 ends on byte-identical weights to SB3 with them in its own process; the boundary costs 0.37-0.58 ms per vector step

2026-10-02 · guangzhao (bridges-venv: torch 2.14.1+cpu, SB3 2.9.0, gymnasium
1.3.0, mujoco 3.14.0), with E48's cross arm running RLinf on the same machine
· protocol: [`PROTOCOL.md`](PROTOCOL.md), after the pilots in
`results/pilot.txt`, before the registered runs · no amendment

---

## The result

SB3's PPO, 16 environments, the same settings and seeds. The arms differ only
in where the environments are stepped:
- **inprocess:** in SB3's own process, in DummyVecEnv;
- **bridge:** in another process, behind plugrl-bridges' `PlugRLVecEnv`,
  over the PlugRL protocol on one connection.

| env | seed | final weights (SHA-256) | episodes | in-process | bridge | per vector step |
| --- | --- | --- | --- | --- | --- | --- |
| Pendulum-v1 | 0 | `ec5b8226...` both | 512, identical | 12.7 s | 16.3 s (+29%) | +0.57 ms |
| | 1 | `062e92e3...` both | 512, identical | 12.7 s | 16.5 s (+29%) | +0.58 ms |
| | 2 | `3a8d968f...` both | 512, identical | 12.7 s | 16.5 s (+29%) | +0.58 ms |
| HalfCheetah-v5 | 0 | `8261c03a...` both | 400, identical | 64.0 s | 76.0 s (+19%) | +0.47 ms |
| | 1 | `86564694...` both | 400, identical | 66.7 s | 76.5 s (+15%) | +0.38 ms |
| | 2 | `65cc8843...` both | 400, identical | 67.0 s | 76.5 s (+14%) | +0.37 ms |

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** all 12 runs completed, and every bridge run's client exited 0.
- **V2:** each bridge run served one connection, with slots 0-15.
- **P1:** 6 of 6 pairs end on the same weights.
- **P2:** 6 of 6 pairs log the same episodes, line for line.
- **P3:** each bridge run's own episode log agrees with its `monitor.csv`.
- **P4:** the bridge adds 0.37-0.58 ms per vector step, inside 0.2-1.0.

---

## What it means

**The boundary is transparent.** Running the environments in another
process, behind the protocol, changed nothing that training sees. It holds
on HalfCheetah too, whose observations are float64 and cross as float32,
because SB3 casts them to float32 before the policy sees them anyway. What
it changes is time.

**What the boundary costs, on one machine.** A vector step costs about
0.4-0.6 ms more. That is one exchange of 16 environments' observations and
actions, packed, sent over the loopback, and unpacked, with a lockstep wait.
It is 14-29% of these runs, because the policies are small MLPs and these
environments step in microseconds. The share falls as the policy grows (see
the paper's VLA paragraph: openpi's forward is 35-100 ms).

**A seed is a machine's, not the code's.** The laptop pilot's Pendulum run
at seed 9 ended on `e8c34d40...` in both arms, while guangzhao's ended on
`3721266c...` in both. Two machines with the same versions give different
weights, as E43 found for MuJoCo. A boundary between them can only be judged
by a status rule, which is why E43, E48 and E49 use one. On one machine it
can be judged byte for byte, which is what E50 does.

---

## What E50 does not show

* **Any trainer but SB3.** CleanRL and RLinf go through the same server and
  share `stack_states` and the episode counting. Neither was run here
  in-process against bridged.
* **Images.** Both environments send states only.
* **Two machines.** By the point above, no byte-for-byte test is possible
  there.
* **A quiet machine.** E48's RLinf run shared guangzhao, at a load of 0.2-3.
  The overheads are consistent across seeds (0.57-0.58 ms on Pendulum), but
  they are not a benchmark.

Kept here: each run's `result.json`, `monitor.csv`, the bridge runs'
`episodes.csv` and logs, `run.out`, `load.txt`, `bridges-src.sha256` and
`verdicts.txt` (`summarise.py`).
