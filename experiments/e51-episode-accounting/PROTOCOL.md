# E51 measurement protocol (pre-registered)

**Written 2026-10-02, after reading the code of every counter below and
before running `audit.py` on any of them.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

A trainer's learning curve is the episode returns some piece of code counted.
That code sits at the boundary: it has to know which step ended an episode
and which step began the next, and the two autoreset conventions put that
step in different places:
- **NEXT_STEP:** the step after an episode ends is a reset step, with no
  reward.
- **SAME_STEP:** the ending step already returns the next episode's first
  observation.

**Which of the episode counters that the trainers in E48-E50 depend on count a
known episode correctly, under each convention that applies to them?**

---

## Declared in advance: what was already known

1. **gymnasium 1.3's vector `RecordEpisodeStatistics`** reads the env's
   `autoreset_mode` in `__init__`, but `step` never uses it. `step` zeroes the
   counters of the envs that ended on the previous step and does not add their
   reward on this step: right for NEXT_STEP, where that step is the reset step.
   Issue 1017 (2024) reported the wrapper counting one step too many under
   NEXT_STEP, and the fix made it count this way.
2. **CleanRL's `ppo_continuous_action.py`** (master; pins gymnasium 0.29.1)
   wraps each sub-env in the single-env `RecordEpisodeStatistics` and logs
   from `infos["final_info"]`, iterating it as a list of per-env dicts.
3. **plugrl-env-client's `VectorEpisodeStatsWrapper`** zeroes an env's
   counters only when `reset` is called for it. Its runner (`rollout.py`)
   calls `reset(options={"reset_indices": done})` after every step that ends
   an episode. The contract is written nowhere.
4. **plugrl-bridges** counts in the adapters (`EpisodeStats`) and, separately,
   in the server's episode log, from the rewards the clients send. Its tests
   cover both.
5. **SB3**'s `DummyVecEnv` resets an env inside the step that ends it, so SB3
   is SAME_STEP by construction. `VecMonitor` and `Monitor` count from the
   rewards they see.

---

## Design

`audit.py` runs every counter on one environment whose episodes are known:
- `Counting-v0`: step t of an episode pays reward t, and the episode
  terminates at t = 5;
- so every episode has return **15** and length **5**;
- dropping the first reward gives 14 and 4, and counting a reset step as a
  step gives length 6.

Two envs, three episodes each, random actions (the env ignores them). Each
counter's logged episodes are compared with (15, 5), and their number with 6.

| id | counter | convention |
| --- | --- | --- |
| A | gymnasium vector `RecordEpisodeStatistics` over `SyncVectorEnv` | NEXT_STEP |
| B | the same | SAME_STEP |
| C | gymnasium single-env `RecordEpisodeStatistics` per sub-env (CleanRL's layering), read from `infos["episode"]` | NEXT_STEP |
| D | the same, read from `infos["final_info"]["episode"]` | SAME_STEP |
| E | CleanRL's logging loop, verbatim, on C's infos | NEXT_STEP |
| F | CleanRL's logging loop, verbatim, on D's infos | SAME_STEP |
| G | SB3 `VecMonitor` over `DummyVecEnv` | (SB3's own) |
| H | SB3 `Monitor` per env inside `DummyVecEnv` | (SB3's own) |
| I | plugrl-bridges `PlugRLVecEnv` behind `VecMonitor`, and the server's episode log, with E50's `gym_client.py` serving `Counting-v0` | SAME_STEP |
| J | plugrl-bridges `PlugRLVectorEnv`'s `infos["episode"]`, the same client | SAME_STEP |
| K | plugrl-env-client `VectorEpisodeStatsWrapper`, driven as `rollout.py` drives it | its runner's |

---

## Predictions, and what falsifies each

**P1.** A counts every episode as (15, 5).

**P2.** B counts the first episode of each env as (15, 5) and every later one
as (14, 4).

> Grounds: known item 1. The first episode has no previous done step.

**P3.** C and D count every episode as (15, 5).

> Grounds: the single-env wrapper sees its own `reset`, which the vector env
> calls in both modes.

**P4.** E logs no episode at all: under gymnasium 1.x's NEXT_STEP there is no
`final_info`. F raises `TypeError`: `final_info` is a dict, iterating it
yields its key strings, `"episode" in "episode"` is true, and indexing a
string with `"episode"` fails.

**P5.** G, H, I (both `VecMonitor` and the episode log), J and K count every
episode as (15, 5).

A prediction fails if any logged episode or count differs from what it says.

**Reported, not predicted:** K's wrapper over gymnasium's `SyncVectorEnv`
with no explicit resets, outside the contract its runner keeps.

---

## Reading order

P1-P5, then the reported case.
