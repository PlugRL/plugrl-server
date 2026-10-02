# E50 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilots in `results/pilot.txt`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E43-E49 show that training through the protocol learns. None of them compares
it with the usual setup, where the trainer steps its environments in its own
process. E46 and E47 show that changes inside the boundary leave the weights
byte-identical, but always with the boundary in place. So:

**On one machine, with the same trainer, settings and seeds, does training
with the environments behind the bridge end on the same weights as training
with them in the trainer's process? And what does the boundary cost?**

---

## Declared in advance: what was already known

1. **The code.**
   - `train.py`: SB3 2.9.0 PPO, 16 envs, 16 x 256 frames per rollout, one
     torch thread, CPU, VecMonitor, `PPO(seed=s)`.
     - The `inprocess` arm uses DummyVecEnv.
     - The `bridge` arm uses plugrl-bridges' `PlugRLVecEnv` at the commit
       synced to guangzhao, with the server's episode log.
   - `gym_client.py`: one connection carries the same 16 environments.
     Env i is reset with seed s + i first and with no seed afterwards, as
     DummyVecEnv does after `seed(s)`.
2. **The envs.**
   - **pendulum:** Pendulum-v1, 102,400 steps, at rl-baselines3-zoo's
     settings (gSDE included).
   - **halfcheetah:** HalfCheetah-v5 (mujoco 3.14.0), 409,600 steps, at SB3's
     defaults with no VecNormalize. Its observations are float64. The client
     sends them as float32, and SB3 casts them to float32 itself.
3. **The pilots, seed 9** (`results/pilot.txt`).
   - **On guangzhao:** both envs ended on byte-identical weights in the two
     arms.
     - pendulum `3721266c...`: 12.7 s against 16.4 s;
     - halfcheetah `0260ca9d...`: 64.3 s against 74.9 s.
     - That is 0.58 and 0.41 ms more per vector step behind the bridge.
   - **On the laptop:** pendulum ended identical in the two arms too, at
     `e8c34d40...`. This is not guangzhao's hash for the same seed: the
     machine changes the weights, and the boundary did not.
4. **The machine is shared.** E48's cross arm runs RLinf on guangzhao
   throughout, at a load of about 1 of 24 cores. The times are reported with
   that noted, and the predictions on time are loose.

---

## Design

On guangzhao, through `run.sh`, one run at a time, with port 8820: env
{pendulum, halfcheetah} x seed {0, 1, 2} x arm {inprocess, bridge}. That is
12 runs.

---

## Checks

* **V1 - every run is complete.** Each `result.json` has `steps` equal to the
  env's budget, and every bridge run's client exited 0.
* **V2 - the bridge served one connection.** Each bridge run's log has one
  `plugrl-bridges: slots 0-15` line.

---

## Predictions, and what falsifies each

**P1 - the boundary leaves the weights alone.** For each env and seed, the
two arms' `weights_sha256` are equal: 6 of 6 pairs.

> Grounds: known item 3. Nothing in the data path differs. The observations
> are float32 in both arms by the time the policy sees them. The rewards are
> rounded to float32 once in both. The actions are the same float32 arrays,
> and the seeds and reset order are the same.

**P2 - and the episodes.** For each pair, the two `monitor.csv` files list the
same returns and lengths, in the same order.

**P3 - the bridge's own count agrees.** In each bridge run, the episode log
lists the same returns (to 1e-4) and lengths as that run's `monitor.csv`.

**P4 - the price on one machine.** The bridge arm takes 0.2 to 1.0 ms more per
vector step than its in-process pair, in every pair. A vector step is one
step of all 16 envs: 6,400 per pendulum run and 25,600 per halfcheetah run.

> Grounds: known item 3, which measured 0.58 and 0.41 ms. E47's 0.96 ms is
> a single env's round trip with HalfCheetah-sized observations.

**Reported, not predicted:**
- each run's learn time and steps per second;
- the overhead as a fraction of the in-process time.

---

## Reading order

V1, V2, P1, P2, P3, P4, then the reported figures.
