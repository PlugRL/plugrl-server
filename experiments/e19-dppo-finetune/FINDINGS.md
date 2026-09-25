# E19: DPPO does not collapse a good policy - and barely moves it

2026-09-25 · Windows 11, CPU only · `fpo-policy × dppo(cheetah) × HalfCheetah-v5`,
three seeds, twenty iterations of 4,096 steps from E16's Phase A checkpoints ·
protocol: [`PROTOCOL.md`](PROTOCOL.md)

---

## The result

E17 and E18 put DPPO in front of a randomly initialised policy, which is not
what it was built for. E19 gave it the job it was built for: the good policies
E16's Phase A left at step 327,680, with the actor and its observation
statistics restored, a fresh value head and a new optimizer. That is E16's
`except-critic` arm with DPPO in FPO's place, and E14's shape.

**Every registered prediction holds.**

| | registered | measured | |
| --- | --- | --- | --- |
| P1(a) | restore logged, 12 critic tensors held back | all three seeds | **holds** |
| P1(b) | first iteration beats random by more than 500 | by 1481 to 2148 | **holds** |
| P3 | under 45 minutes | 9.5 minutes | **holds** |
| P2 | `end − start` above −500 on every seed | −317.3, −118.5, +20.9 | **holds** |

| seed | start | end | `end − start` | E16, FPO, same arm |
| --- | --- | --- | --- | --- |
| 0 | 1438.1 | 1120.8 | −317.3 | +73.0 |
| 1 | 1791.6 | 1673.1 | −118.5 | +366.6 |
| 2 | 1190.8 | 1211.6 | +20.9 | +28.8 |

A trained policy handed to DPPO with a random value head is not destroyed.
The collapse E14 showed is not something DPPO does to a good HalfCheetah
policy.

---

## Why that is a weak statement: DPPO barely touched the policy

Read after the verdicts, from the checkpoints and the training metrics
([`mechanism.py`](mechanism.py), output in
[`results/mechanism.txt`](results/mechanism.txt)):

| seed | actor movement | `approx_kl`, median | `clipfrac`, max |
| --- | --- | --- | --- |
| 0 | 0.0028 | 5.0 × 10⁻⁶ | 0.158 |
| 1 | 0.0043 | 5.3 × 10⁻⁶ | 0.105 |
| 2 | 0.0033 | 5.6 × 10⁻⁶ | 0.097 |

In twenty updates DPPO moved these actors **0.28% to 0.43%** of their norm.
FPO, from the same checkpoints with the same restore mode, moved them **11% to
16%** (E20's `reward` arm, #52): 25 to 49 times as far, seed by seed. A policy that
is hardly changed cannot collapse. So P2 holding says little about how DPPO
treats a good policy, and a lot about how little it changes one in this
configuration.

That is the same finding as E18's, in the setting DPPO was built for. From a
random initialisation `approx_kl` sat near 10⁻⁸ and the clip never engaged.
From a trained policy it is a few 10⁻⁶ and the clip engages on up to 16% of
samples - more, but still two to three orders of magnitude below what a PPO
update on HalfCheetah usually moves. E18's candidate explanation applies
unchanged: the per-step Gaussian whose log-probability DPPO optimises has a
standard deviation of about 0.03, while the initial noise that decides most of
the action has one of 1. That remains a candidate, not a tested cause.

---

## `start` is one iteration, and it shows

`start` is registered as the first iteration alone, because it is the only
measurement of the restored policy under DPPO's noisy sampling before any
update. One iteration on HalfCheetah is noisy, and seed 0's was high: 1438.1,
then 916 after the first update, with the actor having moved almost nothing.
Its whole curve runs from 628 to 1438. Against the mean of the first three
iterations instead - 1082.1, 1826.7, 1160.4 - the change to the last ten is
+38.7, −153.6 and +51.2. Against Phase A's value at the checkpoint, measured
under FPO's deterministic sampling, the last ten sit within 41 of it on every
seed: 1120.8 against 1129.8, 1673.1 against 1632.3, 1211.6 against 1183.0.

None of those is registered, and the protocol says why Phase A's value is not
a fair start. Together they say the same thing as the movement: **the policy
ended where it began.** The −317.3 on seed 0 is mostly a high first
iteration.

P1(b) says one more thing along the way. DPPO collects with
`sampling_noise_level` 0.1 and FPO deterministically, and the cost of that
noise was not known in advance. At the first iteration it is not visible:
1438, 1792 and 1191 under DPPO's sampling against 1130, 1632 and 1183 under
FPO's.

---

## What this does and does not support

**Supported:**

* DPPO can fine-tune from a PlugRL checkpoint (#51), with the actor and its
  statistics restored and the value head fresh, and runs twenty iterations on
  three seeds in under ten minutes.
* In this configuration DPPO does not collapse a good HalfCheetah policy: the
  worst `end − start` was −317.3, against a registered bound of −500, and
  every seed ended within 41 of its Phase A value.
* In this configuration DPPO moves a trained actor by 0.28% to 0.43% in
  twenty updates, 25 to 49 times less than FPO from the same checkpoints.

**Not supported:**

* That DPPO fine-tunes well, or better or worse than FPO. It barely changed
  the policy, and the protocol registered nothing about improvement.
* Anything about pi0.5. E19's actor has 4,390 parameters.
* Why DPPO's updates are this small here. E18's candidate is untested.

Known item 1 of the protocol says every one of E16's nine restarted runs rose.
E20 (#52) has since rerun one of those arms and seen a seed fall by 55.7;
learning is not deterministic here, and single-run gains carry up to about 85
of run-to-run noise. Nothing in E19's verdicts depends on that item.

---

## Reproducing

```bash
bash run.sh            # 3 seeds x 20 iterations, three at a time; needs E16's Phase A
python summarise.py    # P1(a), P1(b), P3, P2
python mechanism.py    # movement, approx_kl, clipfrac, the return curves
```
