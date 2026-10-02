# E49 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilots in `results/pilot.txt`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E48 found that RLinf trains PlugRL env clients through the bridge. One
trainer is one case. The claim is that the boundary does not depend on the
trainer, so:

**Do three trainers that were not written for PlugRL - RLinf, Stable-Baselines3
and CleanRL - train PlugRL env clients through plugrl-bridges, using the
interface each trainer already has? The clients are Python's MuJoCo env
client, and a C++ program that links no third-party library.**

---

## Declared in advance: what was already known

1. **The code.**
   - **plugrl-bridges** at PlugRL/plugrl-rlinf#2 (`79ec044`, with the pilot's
     config fix); the source the
     runs use is recorded in `results/bridges-src.sha256`. Its three
     adapters:
     - `PlugRLEnv` for RLinf (`env_type: plugrl`);
     - `PlugRLVecEnv` for SB3;
     - `PlugRLVectorEnv` for CleanRL, a gymnasium VectorEnv under SAME_STEP
       autoreset.
   - **RLinf** 034579c, with its env-type hunk pointing at
     `plugrl_bridges.rlinf`. Its PPO-MLP runs on one RTX 5090.
   - **SB3 2.9.0 and CleanRL master's `ppo_continuous_action.py`.** The
     CleanRL script changes only its env creation and episode logging. Both
     run in bridges-venv: torch 2.14.1+cpu, gymnasium 1.3.0.
   - **The clients**, always 16 processes of one env each:
     - Pendulum: E44's `pendulum_client`, seeds 100s + 1 to 100s + 16;
     - HalfCheetah-v5: plugrl-env-client main.
   - **`run_bridged.sh TRAINER ENV OUT SEED`** runs one run. The bridge
     writes `episodes.csv` in every run.
2. **The settings.** A rollout is always 16 envs x 256 steps = 4,096 frames.
   - **RLinf x Pendulum:** RLinf's PPO-MLP at rl-zoo's Pendulum settings:
     gamma 0.9, GAE 0.95, 10 epochs, lr 1e-3, clip 0.2, no entropy bonus,
     minibatch 64. **100 epochs, 409,600 steps** (item 4).
   - **SB3 x Pendulum:** rl-zoo's Pendulum-v1 settings with gSDE, 102,400
     steps.
   - **SB3 x HalfCheetah:** SB3's defaults behind VecNormalize, 409,600 steps.
   - **CleanRL x HalfCheetah:** CleanRL's defaults with 16 envs and 256
     steps, 409,600 steps (100 iterations).
3. **CleanRL x Pendulum is not run.** E44's pilot found CleanRL's defaults
   barely learn Pendulum: seed 0 went from -1,194 to -1,149 in 1M steps. E49
   does not tune.
4. **The pilots, seed 9** (`results/pilot.txt`):
   - **SB3 x Pendulum:** -1,254 -> -204 (+1,049). V1-V3 pass.
   - **SB3 x HalfCheetah:** -342 -> 1,171 (+1,514). V1 and V2 pass.
   - **CleanRL x HalfCheetah:** -337 -> 786 (+1,123). V1 and V2 pass.
   - **RLinf x Pendulum, first try:** +7.
     - The config claimed rl-zoo's settings but kept RLinf's CartPole
       minibatch of 1,024, which gives 40 updates a rollout against rl-zoo's
       640. Fixed to 64, and nothing else changed.
     - Second try: +162 over E44's 102,400 steps, below the bar, but climbing
       (its last six batches -985 to -1,150, from -1,267).
     - No defect was found behind the slower learning. RLinf's PPO-MLP differs
       from SB3's (wider layers, value clipping, log-std -0.5, no gSDE) and
       learns Pendulum more slowly at these settings.
     - **Chosen after this pilot, and declared here:** the cell runs 409,600
       steps (E43's budget). Its first 102,400 are reported by E44's rule too.
   - Two checks of `summarise.py` were corrected during the pilots:
     - V2 for RLinf now groups by epoch, as RLinf logs.
     - V3 does not apply under RLinf: RLinf sends no stop (E48), so the C++
       clients never print their count.
5. **E50 and E51 come first.**
   - E50: on one machine, `PlugRLVecEnv` gives SB3 byte-identical weights to
     DummyVecEnv.
   - E51: the adapters' episode counts and the episode log are right.

---

## Design

The four cells of item 2, seeds 0, 1 and 2: 12 runs, one at a time, on
guangzhao, on port 8810 (`run_e49.sh`).

A **batch** is the k-th episode of every slot. All 16 envs start together and
their episodes have a fixed length (200 and 1,000 steps), so a batch ends on
one step. A run has:
- 32 Pendulum batches;
- 25 HalfCheetah batches;
- 128 batches for RLinf x Pendulum.

---

## Checks

* **V1 - the run is complete.**
  - The trainer exits 0, and its log has no `Traceback`, `ClientLost` or
    `TimeoutError`.
  - `episodes.csv` holds every slot's full count of episodes, each of the
    env's length.
* **V2 - the trainer saw the bridge's episodes.** The trainer's own record
  matches the bridge's log, to 1e-4 relative plus 0.01:
  - SB3's `monitor.csv` holds the same returns;
  - CleanRL's `charts/episodic_return` holds the same returns;
  - RLinf's `env/return` equals each batch's mean.
* **V3 - SB3 x Pendulum only: the C++ clients agree.** Each client reports
  6,400 steps and 32 episodes. Its own mean over its last 10 episodes matches
  a slot's in the log, to 0.01. V3 does not apply under RLinf (item 4).

---

## Predictions, and what falsifies each

The status rules are fixed per env:
- **Pendulum (E44's):** the mean of batches 20-32 is at least **500** above
  the mean of batches 1-2.
  - For RLinf x Pendulum's 128 batches, E44's windows scale. Batches 1-2 are
    the first 8% of its budget, as in E44, and batches 77-128 are the last
    40%. The bar is still 500.
- **HalfCheetah (E43's and E48's):** the mean of batches 21-25 is at least
  **200** above batch 1.

**P1 - RLinf x Pendulum learns on at least 2 of 3 seeds.**

> Grounds, and the weakest of the four: the second pilot climbed about 6 a
> batch over its 32 batches. If that held, batches 77-128 would sit about
> 600 above batches 1-2. A Pendulum curve is rarely straight, and no pilot
> ran 409,600 steps.

**P2 - SB3 x Pendulum learns on at least 2 of 3 seeds.**

**P3 - SB3 x HalfCheetah learns on at least 2 of 3 seeds.**

**P4 - CleanRL x HalfCheetah learns on at least 2 of 3 seeds.**

> Grounds for P2-P4: each pilot cleared its bar at least twice over
> (+1,049 against 500; +1,514 and +1,123 against 200).

A run that fails V1 is not judged by its status rule; it counts as not
learning. That is a defect to find, not a finding about the trainer.

**Reported, not predicted:**
- RLinf x Pendulum's gain over its first 102,400 steps, by E44's rule;
- every run's batch means;
- the wall clock of each run;
- the final batch means against E44 (Pendulum, -251 to -206) and E48
  (HalfCheetah, RLinf).

---

## Declared deviations allowed in advance

1. A run that dies for a reason outside the experiment, such as guangzhao's
   GPU being taken, is rerun once. This is recorded in `AMENDMENT.md`.

---

## Reading order

V1, V2, V3, P1-P4, then the reported figures.
