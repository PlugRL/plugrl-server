# E30: under DPPO, `fpo-policy`'s network is what stops it; with all of `dppo-policy`'s structure it learns Hopper

2026-09-27 · Linux workstation (`guangzhao`), CPU only · five arms on
Hopper-v5, three seeds each, 100 iterations of 4,096 steps · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`c5d3ce4`, after the pilot in
`results/pilot.txt`, before the registered run)

---

## The result

E28 found `fpo-policy` barely moves under DPPO where `dppo-policy` learns;
E29 ruled out DPPO's log-probability clamp and found ten times the noise
reaches about a quarter of `dppo-policy`'s gain. E30 took the three
remaining structural differences between the two policies one at a time, on
top of E29's noise level, and all three together.

| arm | on top of E29's `wide` | gain, seeds 0 / 1 / 2 | closes most of the gap (+200 on 2 of 3) | iterations 91-100 |
| --- | --- | --- | --- | --- |
| `base` | nothing | +73 / +89 / +106 | no | 90 / 110 / 126 |
| `chunk4` | 4 actions per inference | +48 / +60 / +45 | no | 68 / 78 / 63 |
| `steps20` | 20 flow steps | +64 / +85 / +100 | no | 81 / 108 / 121 |
| `net512` | 3 hidden layers of 512 | **+334 / +334 / +339** | **yes** | 352 / 352 / 357 |
| `all` | all three | **+742 / +853 / +968** | **yes** | **762 / 873 / 989** |

* **P1 holds**: all fifteen seeds ran 100 iterations, wrote the checkpoint
  at 100, have no traceback and a client that exited 0; 8 to 36 minutes per
  arm.
* **V1 passes** for every arm: the logged entropy sat at +0.3828
  (expected +0.383) for `base`, `chunk4` and `net512`, and at -0.0040 for
  `steps20` and `all`, at every iteration.
* **P2 holds**: the network closes most of the gap on its own.
* **P3 holds**: neither the chunk nor the step count does.
* **Reported**: `all` meets the coverage figure's Hopper bar - iterations
  91-100 at least 500 - on all three seeds, and no other arm does on any.

`base` repeats E29's `wide` arm: +73 / +89 / +106 here, +74 / +85 / +106
there.

---

## The curves

Mean return over windows of ten iterations (the first: iteration 1 alone):

| arm, seed | 1 | 11-20 | 21-30 | 41-50 | 61-70 | 81-90 | 91-100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `net512`, 0 | 18 | 101 | 166 | 228 | 295 | 345 | 352 |
| `net512`, 1 | 18 | 100 | 170 | 223 | 281 | 342 | 352 |
| `net512`, 2 | 17 | 112 | 189 | 234 | 322 | 345 | 357 |
| `all`, 0 | 20 | 62 | 148 | 251 | 347 | 497 | 762 |
| `all`, 1 | 20 | 59 | 130 | 249 | 294 | 631 | 873 |
| `all`, 2 | 21 | 64 | 135 | 285 | 556 | 739 | 989 |
| `base`, 0 | 17 | 18 | 19 | 26 | 57 | 80 | 90 |

`net512` rises fastest early and levels off near 350 on every seed, with
episodes of 132 to 138 steps. `all` starts slower, passes `net512` by
iterations 41-50 and is still climbing steeply at the end: its episodes run
627 to 909 steps. The other three arms gain at most 106.

The updates say the same thing as the pilot did. Over iterations 1-10,
`approx_kl` averaged 1.0 x 10⁻⁵ for `net512` and 0.9 x 10⁻⁸ to 5.4 x 10⁻⁷
for the others; `clipfrac` 1.4% for `net512`, at most 0.013% elsewhere.
`dppo-policy` in E28 ran at about 10⁻⁵ and 1-2%. At the same noise, the
small network's updates are about 400 times smaller in KL than the wide
one's (`base` against `net512`).

---

## What it means for the coverage figure

`fpo-policy` under DPPO was not learning because it ran with the structure
FPO's playground gives it - four layers of 32, one action per inference, ten
flow steps - and the noise level of FPO's defaults. With DPPO's own MuJoCo
structure, the one `dppo-policy` carries (three residual layers of 512,
chunks of 4, 20 steps) and noise matched to its entropy, it learns Hopper in
100 iterations, where `dppo-policy` needed about 150 to 200 in E31.

That configuration was chosen here, from a screen on Hopper, after seeing
Hopper. It is not yet evidence for the other two tasks, or for Hopper on
seeds the choice was made with: the next experiment registers `all` as it
stands and runs it on HalfCheetah, Hopper and Walker2d.

---

## What this does not show

* Why the network matters. `net512` changes width, depth and the residual
  connections at once, and the MLP's output scale of 0.25 stayed as it was.
* Why `chunk4` and `steps20` help only together with the network. Alone,
  each is within the others' range; `all` gains 2.5 times `net512`.
* Anything at FPO's default noise level; every arm ran at 1.0.
* Other tasks.

---

## Files

* `results/run.out` - the launcher's log: code, client, per-arm failed-seed
  counts (all 0).
* `results/<arm>.out`, `results/<arm>/{server,client}-seed<N>.log` - each
  arm's launcher output and each seed's server and client log.
* `results/verdicts.txt` - `summarise.py`'s output, run on `guangzhao`
  against the tensorboards and checkpoints, which stay there
  (`~/zuogou/plugrl/e30/experiments/e30-fpo-dppo-factors/results/<arm>/dppo/`).
* `results/pilot.txt` - the eleven-iteration pilot, before the protocol.
* `summary.tsv` - first and last-ten return and episode length per seed.
