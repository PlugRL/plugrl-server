# E44 measurement protocol (pre-registered)

**Written 2026-09-28, after the pilot in `results/pilot.txt`, before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

PlugRL's home page says an env client does not have to be Python. E2 showed
a C++ client with no third-party libraries completing 120 exchanges with a
real server, on invented observations and rewards. Nothing has yet been
trained through a non-Python client.

**Does a C++ program with no third-party libraries, stepping an environment
of its own, train a policy through PlugRL as well as the Python env client
does?**

---

## Declared in advance: what was already known

1. **The client** (`pendulum_client.cpp`).
   - Up to its WebSocket class it is plugrl-protocol's
     `examples/plugrl_client.cpp` at 3a59528, unchanged except where marked
     "E44": a close now carries its reason, and an episode's statistics go in
     `info`.
   - After that it has gymnasium's Pendulum-v1 written out, and a loop that
     does what plugrl-env-client's rollout does with one env and replan 1.
   - Built with g++ 11.4 at -O2, it links only libstdc++, libm, libgcc and
     libc.
2. **The environment is gymnasium's.** Across 20 trials of 200 steps, from
   random states under random torques (some outside the clip):
   - the observations were bit-identical;
   - the rewards agreed within 1.2e-10, below float32's resolution, and the
     reward travels as float32;
   - the time limit fell on the same step every time.
3. **A gap in SPEC.md, found by the pilot.** The server's episode metrics
   read `info["episode"]` (r, l, s, mask), and SPEC.md does not document it.
   - The client, written from the spec, reported no episodes at all.
   - It now sends them as plugrl-env-client's wrapper does.
   - Training reads only the feedback's reward and flags, so this changes the
     metrics and not the learning.
4. **The control** is plugrl-env-client b20f6bd with the Classic-v1 fix (PR
   #10), stepping gymnasium 1.3.0's Pendulum-v1, with the same server.
5. **The pilot's learning**, seed 0.
   - **CleanRL's default PPO settings barely learn Pendulum**, in both arms
     alike: -1194 -> -1149 (cpp) and -1196 -> -1011 (python) over its own
     1M steps. So the registered run does not use them.
   - **rl-baselines3-zoo's PPO settings for Pendulum-v1 learn it in both
     arms**: -1255 -> -208 (cpp) and -1137 -> -249 (python) over iterations
     1-2 -> 16-25. rl-zoo reports -172 ± 104.
   - Those settings are rl-zoo's published ones (`hyperparams/ppo.yml`, last
     changed at eec15dc), not tuned here: gamma 0.9, gae_lambda 0.95, 10
     epochs, lr 1e-3 constant, clip 0.2, ent 0, no normalisation, 1e5 steps.
   - Two things differ from rl-zoo:
     - one env collecting 4,096 steps an iteration, where rl-zoo uses 4 x
       1,024;
     - no gSDE, which gaussian-policy has no counterpart for.

---

## Design

Two arms, run at the same time on guangzhao, seeds 1-3 (the pilot used seed
0):

- **cpp**: the C++ client.
- **python**: the control.

Both train gaussian-policy (obs 3, action 1, `--policy.action-clip 2.0`)
with ppo and these `PPO_ARGS`:

```
--policy.no-normalize-observations --algo.buffer-size 4096
--algo.batch-size 64 --algo.update-epochs 10 --algo.gamma 0.9
--algo.gae-lambda 0.95 --algo.learning-rate 1e-3 --algo.no-anneal-lr
--algo.clip-coef 0.2 --algo.no-clip-vloss --algo.ent-coef 0.0
--algo.no-normalize-rewards
```

The run is 25 iterations, 102,400 steps. The server seed initialises the
policy, and each client seeds its own environment. Launched by `run.sh`,
with every earlier run's port closed first.

---

## Checks

* **V1 - the arms are what they say.**
  - Every cpp server's client is the C++ binary; every python server's is
    plugrl-env-client.
  - All six server configurations are identical apart from port and name,
    and each shows the settings above.

---

## The status rule

An arm **learns** if, on at least 2 of 3 seeds, the mean return over
iterations 16-25 is at least **+500** above its mean over iterations 1-2.

That is half the distance from a random policy (about -1,200) to rl-zoo's
-172. The pilot's gains were +1,047 (cpp) and +888 (python).

---

## Predictions, and what falsifies each

**P1 - both arms run end to end.** On every seed: 25 iterations logged, no
traceback, and every client exits 0.

**P2 - the C++ arm learns.** **P3 - the control learns.**

> Grounds: known items 2 and 5. Each is falsified if fewer than two seeds
> clear the bar.

**P4 - the C++ arm learns as the control does.** Every C++ seed's mean
return over iterations 16-25 lies within the control's three seeds' range,
widened by **150** on each side.

> Grounds: the environments are identical (item 2). What differs is who
> steps them and how the feedback is packed. A mistake in the packing - a
> misattributed reward, or an episode boundary one step off - would pull the
> C++ arm below the control. So this checks the C++ client's use of the
> protocol, not only its physics. The 150 is rl-zoo's per-episode spread,
> rounded up; the pilot's two arms ended 41 apart.

**Reported, not predicted:**
- the returns at iterations 1-2, 10 and 16-25 for every seed
- wall clock per arm
- the C++ client's own count of steps and episodes, and its last-ten-episode
  mean, against the server's

---

## Declared deviations allowed in advance

1. One restart of any seed that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, P3, P4, then the reported figures.
