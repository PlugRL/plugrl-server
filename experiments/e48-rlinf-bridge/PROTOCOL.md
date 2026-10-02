# E48 measurement protocol (pre-registered)

**Written 2026-10-01, after the pilot in `results/pilot.txt`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

Every PlugRL result so far was trained by plugrl-server. The project's
claim is about the protocol: an environment that speaks it can be trained
by any trainer that does. RLinf is a second trainer, independent of this
codebase. Its environments normally run inside its own Ray cluster, in its
own Python, with its own torch, on every node. So:

**Does RLinf, through the bridge, train environments that PlugRL env clients
run outside its cluster: as other processes on its machine, and on another
machine that has no torch, no Ray and no RLinf?**

---

## Declared in advance: what was already known

1. **The code** (`results/pilot.txt`).
   - **The bridge:** PlugRL/plugrl-rlinf at fca69db. `PlugRLEnv` is an RLinf
     environment, and `BridgeServer` is a lockstep PlugRL server inside
     RLinf's environment worker.
   - **RLinf:** 034579c plus the bridge's two-hunk patch, installed by its
     own `install.sh`, with torch 2.11.0+cu130 and ray 2.58.0.
   - **The env clients:** plugrl-env-client main, 16 processes of one
     HalfCheetah-v5 env each.
     - On guangzhao: mujoco 3.14.0, gymnasium 1.3.0.
     - On the laptop: mujoco 3.6.0, gymnasium 1.2.3, in a venv with no
       torch, ray or rlinf, 276 MB in all.
   - **The config:** RLinf's own CartPole PPO-MLP config with the env swapped,
     16 envs × 256 steps = 4,096 frames per epoch, 100 epochs, 409,600 steps
     (E43's budget), one RTX 5090.
2. **The pilot.**
   - Locally, the return of the first five episode batches rose from -267
     to -110 in 20 epochs, at 3.6-4.6 s per epoch.
   - Across machines, 5 epochs took 22.9-24.3 s each over a 49-54 ms link,
     and the per-step reward rose from -0.34 to -0.22.
   - Four defects were found and fixed before this file (pilot.txt). The
     worst was a reconnect storm: 2 million connections from leftover
     client processes.
3. **The physics differ across the two machines.** The mujoco and gymnasium
   versions are not the same, so the two arms are compared by the status
   rule, not step by step.
4. **The link varies.** It measured 3 ms and 54 ms on the same day. The
   cross arm's time per epoch is reported, not predicted.

---

## Design

Six runs, one at a time, all with RLinf on guangzhao. `run_e48.sh` runs them
from the laptop:
- **local, seeds 0, 1, 2:** RLinf and the 16 env clients both on guangzhao
  (`run_guangzhao.sh OUT SEED local`);
- **cross, seeds 0, 1, 2:** RLinf on guangzhao, the 16 env clients on the
  laptop over Tailscale (`run_guangzhao.sh OUT SEED remote` and
  `cross_clients.py`).

The seed seeds RLinf's actor and the env clients. Episodes are 1,000 steps
(HalfCheetah's time limit), so all 16 envs end an episode together about
every 4 epochs, and RLinf logs `env/return` once per batch: 25 batches in
100 epochs.

---

## Checks

* **V1 - the envs are where they should be.** Each run's log has 16
  `plugrl-rlinf: slots` lines. They come from 127.0.0.1 in the local arm,
  and from the laptop's Tailscale address, 100.65.224.34, in the cross arm.
* **V2 - the run is complete.** RLinf logs 100 epochs, and its log has no
  `Traceback`, `ClientLost` or `TimeoutError`.

---

## Predictions, and what falsifies each

The status rule is E43's: a run **learns** if the mean `env/return` of its
last five episode batches is at least **200** above its first batch's.

**P1 - RLinf learns through the bridge, with the clients on its machine.**
The local arm learns on at least 2 of 3 seeds.

> Grounds: known item 2. The pilot rose 157 in four batches.

**P2 - and with the clients on another machine.** The cross arm learns on at
least 2 of 3 seeds.

> Grounds: as P1. Where the environment is stepped changes the timing, not
> the data a step produces (E43). The physics versions differ (known item
> 3), which is why this is a status rule and not a comparison of curves.

**Reported, not predicted:**
- every run's `env/return` curve;
- the median time per epoch in each arm;
- the link's ping before and after each cross run;
- the wall clock;
- the slot lines.

---

## Declared deviations allowed in advance

1. A cross run that dies for a reason outside the experiment, such as the
   laptop's Wi-Fi dropping, is rerun once. This is recorded in `AMENDMENT.md`.

---

## Reading order

V1, V2, P1, P2, then the reported figures.
