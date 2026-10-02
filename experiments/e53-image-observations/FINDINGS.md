# E53: with camera frames and a CNN, SB3 still ends on byte-identical weights through the bridge; 16 frames a step cost 2.6-3.0 ms

2026-10-02 · guangzhao (bridges-venv: torch 2.14.1+cpu, SB3 2.9.0, gymnasium
1.3.0, ale-py 0.12.1, opencv-python-headless 5.0.0.93; plugrl-bridges#3,
803141d) · protocol: [`PROTOCOL.md`](PROTOCOL.md) (`32af99b`, after the pilot
in `results/pilot.txt`, before the registered runs) · no amendment

---

## The result

SB3's PPO with CnnPolicy trained 16 `AtariWrapper(ALE/Pong-v5)` environments
for 40,960 steps, with 84x84 grayscale frames, stacked 4 deep. The two arms:
- **inprocess:** in SB3's DummyVecEnv;
- **bridge:** in another process, behind
  `PlugRLVecEnv(image_key="frame")`, which receives each uint8 frame as
  rendered.

| seed | final weights (SHA-256) | episodes | in-process | bridge | per vector step |
| --- | --- | --- | --- | --- | --- |
| 0 | `82f80c02...` both | 183, identical | 67.4 s | 75.0 s (+11%) | +2.99 ms |
| 1 | `60acba42...` both | 176, identical | 69.6 s | 76.2 s (+10%) | +2.60 ms |
| 2 | `6bf04095...` both | 183, identical | 68.1 s | 75.8 s (+11%) | +2.98 ms |

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** all 6 runs are complete. Each bridge run served one connection, and
  its observation space was `Box(0, 255, (84, 84, 4), uint8)`, as in process.
- **P1:** 3 of 3 seeds end on the same weights.
- **P2:** both arms logged the same episodes, and the bridge's own episode
  log agrees. (AtariWrapper ends an episode at each lost life.)
- **P3:** the bridge adds 2.60-2.99 ms per vector step, inside 1.0-5.0.

---

## What it means

- **The boundary stays transparent with images.** E50 showed this for state
  vectors. Here the observations are camera frames: a uint8 image space, a
  CNN, SB3's own transposition and scaling, and frame stacking on the
  trainer's side. Nothing in that path changes when the frames cross a
  process boundary.
- **Images cost more to cross, in proportion to their bytes.** Each step
  moves 16 frames of 7 KiB twice: in the feedback, and again in the next
  infer, since this client does not use `reuse-feedback-obs`. That is about
  2.8 ms on the loopback, against 0.4-0.6 ms for E50's states. It is still
  10-11% of a short CNN run on CPU, and a far smaller share beside a VLA's
  forward pass (E10).

---

## What E53 does not show

* **Learning.** 40,960 steps is far too short to learn Pong. E53 compares
  weights. E49 shows learning through the bridge, with state observations.
* **A GPU.** The CNN ran on CPU with four threads, in both arms. On a GPU,
  bit-identical comparisons need deterministic kernels.
* **Two machines, or reuse.** E43 and E46 measured what frames cost across a
  link, and how `reuse-feedback-obs` halves it.

Kept here:
- each run's `result.json`, `monitor.csv`, the bridge runs' `episodes.csv`,
  and the logs;
- `registered.out`, `load.txt`, `bridges-src.sha256` and `verdicts.txt`
  (`summarise.py`).
