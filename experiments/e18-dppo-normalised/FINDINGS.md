# E18: normalising DPPO's inputs helps a little, and does not make it learn

2026-09-25 · Windows 11, CPU only · `fpo-policy × dppo(cheetah) × HalfCheetah-v5`,
three seeds, 409,600 steps each · protocol: [`PROTOCOL.md`](PROTOCOL.md) ·
amendments: [`AMENDMENT.md`](AMENDMENT.md) - 1, the machine was shared; 2,
training is not deterministic

---

## The result

E17 ran DPPO on `fpo-policy` with the policy's observation statistics never
initialised - `obs_stats_count` at 0.0, the identity - on inputs whose
dimensions differ in scale by 59x. PR #50 fixed it. E18 reran E17's design on
the fixed code and asked whether DPPO then learns.

**It does not.** P2 asked for `end − start` above +200 on two of three seeds:

| seed | start | best | end | E18 gain | E17 gain, same seed |
| --- | --- | --- | --- | --- | --- |
| 0 | −298.8 | −274.0 | −293.6 | **+5.2** | −21.2 |
| 1 | −319.9 | −297.2 | −299.7 | **+20.2** | −8.4 |
| 2 | −312.2 | −258.6 | −309.8 | **+2.4** | −28.5 |

**P2 is falsified, 0 of 3.**

**The fix did something, and how much can be measured without trusting any
run to reproduce another.** A draft of this document read the right-hand two
columns as a paired comparison - each E18 seed against the E17 seed of the same
number, differing only in the code - because their first iterations matched to
the decimal. That was an overclaim, and E20 (#52) is what showed it: run one
configuration twice and **collection is deterministic but learning is not**.
The two runs agree until the first update and then drift apart - by 21% after a
single update on one seed. So the per-seed differences, +26.4, +28.6 and +30.9,
include run-to-run noise of unknown size and cannot be read as the fix alone.

Read as two independent groups of three instead, the fix is still visible:

| | mean gain | sd |
| --- | --- | --- |
| E17, statistics never initialised | −19.4 | 10.2 |
| E18, statistics maintained | +9.3 | 9.6 |

A difference of **+28.6**, Welch t = 3.55 on 4.0 degrees of freedom,
**two-sided p = 0.024**. Modest evidence from three seeds a side, and it depends
on nothing reproducing anything. Normalising the inputs is worth roughly thirty
on this scale. The threshold was two hundred, and FPO on the same line covers
two thousand.

The observation statistics were a real defect, and fixing it is measurably
better. It is not why DPPO does not learn here.

---

## P1: the fix was live in the run

At the final checkpoint of every seed:

| seed | `obs_stats_count` | std, smallest to largest | spread |
| --- | --- | --- | --- |
| 0 | 409,600 | 0.22 to 7.66 | 34.5x |
| 1 | 409,600 | 0.23 to 7.64 | 33.7x |
| 2 | 409,600 | 0.23 to 7.63 | 33.7x |

Against 0.0 and exactly 1.0 in E17. The count is every observation the run
saw, once each. And E18's first iteration matched E17's to the decimal on all
three seeds, which is what a fix that updates the statistics *after* each learn
should do: the first collection happens before any learn, under the identity,
in both. The spread is below FPO's 59x because a policy that never
leaves the neighbourhood of random never reaches the joint velocities a
running cheetah does. **P1 holds, 3 of 3**, so P2 measured what it was built
to measure.

P3, the wall clock, is not read: from 11:47 a CPU-heavy job from a separate
session on another project took about eleven cores (amendment 1), so the
elapsed time is not a cost measurement. Whether the sharing changed the data
as well cannot be ruled out, because learning here is not deterministic
(amendment 2); neither verdict above is close enough to its threshold for
that to matter.

---

## Why it does not learn: the policy barely changes

Read from seed 0's metrics after the fact, and reported as mechanism rather
than as a registered result:

| metric | value | what it means |
| --- | --- | --- |
| `losses/approx_kl` | 2 to 4 × 10⁻⁸, at most 1.8 × 10⁻⁷ | the action distribution moves by almost nothing per iteration |
| `losses/clipfrac` | **0 throughout** | the 0.01 clip never binds; no ratio leaves [0.99, 1.01] |
| `rollout/explained_variance` | −0.0004 rising to 0.53 | the critic learns |
| `models/actor_learning_rate` | 1e-4, constant | the variant's `min_lr` equals `actor_lr`, so nothing anneals |

A PPO-style update on HalfCheetah typically moves the policy by an
`approx_kl` of 10⁻³ to 10⁻². This moves it four to five orders of magnitude
less. The value head learns to predict returns and the policy does not use
the prediction, because each update hardly changes what it does.

**A candidate, not established.** DPPO's log-probabilities are the densities
of each denoising step's Gaussian transition, whose standard deviation at
`sampling_noise_level` 0.1 is about 0.03. Almost all of the final action's
variation comes instead from the initial noise, which has standard deviation
one. So the part of the action the per-step log-probabilities can see is a
small fraction of the part that decides the return, and the gradient that
reaches the actor carries little of the credit. DPPO's paper fine-tunes only
the last few denoising steps and keeps a floor under the noise; this
implementation denoises all ten under the same level. Whether that is the
reason, or whether something in how PlugRL wires DPPO is, E18 does not say.
By the project's rule it is ours until shown otherwise.

---

## What this does and does not support

**Supported:**

* Maintaining the policy's observation statistics makes DPPO measurably
  better on this task, by about thirty in mean return over 409,600 steps -
  Welch t = 3.55, 4.0 degrees of freedom, p = 0.024, three seeds a side. As an
  unpaired comparison: learning is not deterministic run to run (E20), so the
  per-seed pairing E18 first claimed does not hold.
* It does not make DPPO learn HalfCheetah from a random initialisation in this
  configuration. No seed gained more than 20.2.
* In this configuration DPPO's updates change the policy's action
  distribution by an `approx_kl` near 10⁻⁸ per iteration, and its clip never
  engages.

**Not supported:**

* Anything about DPPO as an algorithm. The configuration is its fine-tuning
  variant, run from scratch, on a buffer a fifth of its default size.
* That the per-step noise is the reason the updates are small. It is a
  candidate that fits the numbers and has not been tested.
* Any comparison with FPO.

---

## What comes next

E19 puts DPPO in the setting it was built for - fine-tuning the good policies
E16 left - which bears directly on the same question: if updates this small
are intrinsic to this configuration, a good policy should come out of E19
barely changed, neither improved nor damaged.

## Reproducing

```bash
bash run.sh            # 3 seeds x 409,600 steps, three at a time
python summarise.py    # P1 from the checkpoints, P2 from the curves
```
