# E53 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilot in `results/pilot.txt`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E50 found the boundary transparent on one machine: SB3 ends on byte-identical
weights with its environments in its own process or behind the bridge. But
only state vectors crossed. Image policies are where the boundary matters
most (VLAs; E43's and E46's costs), and a camera frame takes a different
path: uint8 arrays, an image observation space, a CNN, frame stacking. So:

**With image observations and a CNN policy, does SB3 still end on the same
weights through the bridge? And what does crossing cost per step when every
step carries 16 frames?**

---

## Declared in advance: what was already known

1. **The code.**
   - `train.py` follows E50's design. SB3 2.9.0 PPO runs CnnPolicy at
     rl-baselines3-zoo's Atari settings, held constant: n_steps 128,
     4 epochs, minibatch 256, lr 2.5e-4, clip 0.1, entropy 0.01, value 0.5.
   - There are 16 `AtariWrapper(gym.make("ALE/Pong-v5"))` environments,
     with VecMonitor and VecFrameStack(4) on both arms.
   - The run is 40,960 steps on CPU, four torch threads.
   - The bridge arm uses plugrl-bridges#3 (`803141d`):
     `PlugRLVecEnv(image_key="frame")`.
   - `atari_client.py` sends each 84x84x1 uint8 frame as rendered. It
     applies the float action that crosses as an integer, and it seeds and
     resets as DummyVecEnv does.
2. **The versions.** ale-py 0.12.1 and opencv-python-headless 5.0.0.93 were
   installed into bridges-venv for E53.
3. **The pilot, seed 9** (`results/pilot.txt`).
   - Both arms ended on `76f991b8...`, with the observation space
     `Box(0, 255, (84, 84, 4), uint8)` in both.
   - The arms took 68.3 s in process and 74.5 s through the bridge. That is
     2.44 ms more per vector step (2,560 of them).
   - An earlier attempt failed for want of OpenCV. Its stuck bridge process
     then took a later attempt's client, and both were discarded
     (`pilot.txt`).
4. **This is far too short to learn Pong.** E53 compares weights, not
   returns.

---

## Design

On guangzhao, one run at a time, port 8830: seeds 0, 1 and 2, each with arms
inprocess and bridge. That is 6 runs (`run.sh`). The port is checked free
before each run.

---

## Checks

* **V1 - complete.** Every run has 40,960 steps. Every bridge run's client
  exited 0, its log has one `plugrl-bridges: slots 0-15` line, and its
  observation space is uint8.

---

## Predictions, and what falsifies each

**P1 - the boundary leaves the weights alone, with images too.** For each
seed, the two arms' weights are equal: 3 of 3.

> Grounds: the pilot, and E50. Nothing in the data path differs. The frames
> are the same uint8 arrays, and SB3 applies its own image preprocessing
> (transpose, /255) in both arms alike.

**P2 - and the episodes.** For each seed, the two `monitor.csv` files list
the same episodes, and the bridge's episode log agrees with the bridge
arm's.

**P3 - the price of 16 frames a step, on one machine.** The bridge arm takes
1.0 to 5.0 ms more per vector step than its in-process pair, on every seed.

> Grounds: the pilot's 2.44 ms. E50's state-only steps cost 0.37-0.58 ms.
> Each step now moves 16 x 7 KiB of frames twice, in the feedback and again
> in the next infer, since this client does not reuse observations.

**Reported, not predicted:** each run's learn time and steps per second, and
the overhead as a fraction of the in-process time.

---

## Reading order

V1, P1, P2, P3, then the reported figures.
