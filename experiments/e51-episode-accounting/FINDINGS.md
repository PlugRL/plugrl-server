# E51: under SAME_STEP autoreset, gymnasium's vector RecordEpisodeStatistics drops each episode's first reward, and CleanRL's logging logs nothing under gymnasium 1.x's default

2026-10-02 · laptop, gymnasium 1.3.0, numpy 2.3.2, SB3 2.9.0, plugrl-bridges
(source synced to guangzhao for E50) · protocol: [`PROTOCOL.md`](PROTOCOL.md),
written after reading each counter's code and before running any · no
amendment

---

## The result

Every counter ran on `Counting-v0`, which pays reward t at step t and ends at
t = 5. Every episode is therefore (return 15, length 5). Two envs ran three
episodes each (`results/audit.txt`).

| id | counter | autoreset | logged |
| --- | --- | --- | --- |
| A | gymnasium vector `RecordEpisodeStatistics` | NEXT_STEP | 6 x (15, 5) |
| **B** | the same | **SAME_STEP** | 2 x (15, 5), then **4 x (14, 4)** |
| C | single-env `RecordEpisodeStatistics` per sub-env (CleanRL's layering) | NEXT_STEP | 6 x (15, 5) |
| D | the same, from `final_info` | SAME_STEP | 6 x (15, 5) |
| **E** | CleanRL's logging loop, verbatim | NEXT_STEP | **nothing** |
| **F** | the same | SAME_STEP | **`TypeError`** |
| G | SB3 `VecMonitor` over `DummyVecEnv` | SB3's | 6 x (15, 5) |
| H | SB3 `Monitor` per env | SB3's | 6 x (15, 5) |
| I | plugrl-bridges `PlugRLVecEnv` + `VecMonitor`, and the episode log | SAME_STEP | 6 x (15, 5), both |
| J | plugrl-bridges `PlugRLVectorEnv`, and the episode log | SAME_STEP | 6 x (15, 5), both |
| K | plugrl-env-client `VectorEpisodeStatsWrapper`, as its runner drives it | its runner's | 6 x (15, 5) |

All five predictions hold:
- **P1 holds.** A is right under NEXT_STEP.
- **P2 holds.** B is right only for each env's first episode. After that, every
  episode loses its first reward and one step, and nothing warns.
- **P3 holds.** C and D are right.
- **P4 holds.** Under gymnasium 1.x's default (NEXT_STEP), E finds no
  `final_info` and logs no episode, without an error. Under SAME_STEP, F fails
  as predicted: iterating the dict gives its keys, `"episode" in "episode"` is
  true, and indexing a string with it raises.
- **P5 holds.** G-K are right.

**Reported:** plugrl-env-client's wrapper, put over gymnasium's
`SyncVectorEnv` with nobody calling `reset` for the envs that ended, logged
(15, 5), then (30, 11), then (45, 17). Its counters are zeroed only by an
explicit `reset`. Its own runner always calls one, which is why K is right.
That contract is written nowhere.

---

## What it means

- **The wrong count is quiet and small.** B logs every episode but the first
  14/15 of its return and one step short. On a dense task the curve barely
  moves. On a task whose reward comes on the first step, the curve is wrong
  by the whole reward. Nothing in gymnasium's output says which convention
  the counter assumed. It reads `autoreset_mode` in `__init__` and never uses
  it.
- **It has happened before, the other way round.** Issue 1017 (2024) reported
  this wrapper counting one step too many. At the time the vector envs reset
  on the step after the end. The fix made it count for NEXT_STEP. SAME_STEP,
  which gymnasium added later, now gets the opposite error.
- **CleanRL is right on the version it pins** (gymnasium 0.29.1). The failure
  is version skew. Its loop expects 0.29's list of per-env dicts. 1.x's default
  gives it nothing to iterate, and SAME_STEP gives it a dict.
- **The counts this project's results rest on are right.** That covers SB3's,
  the bridge's adapters and episode log, and the env client's wrapper as its
  runner uses it.

---

## What E51 does not show

* **RLinf's own counters.** Through the bridge, RLinf's episode figures come
  from the adapter (I and J cover the shared code). RLinf's own environments
  were not audited.
* **Any counter not listed.** TorchRL, Tianshou, skrl and RLlib were not run.
* **Truncation.** `Counting-v0` terminates. A time-limit truncation goes
  through the same done flag in every counter here, but it was not run.

The upstream report to gymnasium has not been filed.
