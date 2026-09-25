# E20: a random critic's noise does not collapse a good HalfCheetah policy

2026-09-25 · Windows 11, CPU only · `fpo-policy × fpo × HalfCheetah-v5`,
three seeds, 81,920 steps each from E16's Phase A checkpoints · protocol:
[`PROTOCOL.md`](PROTOCOL.md) · amendment: [`AMENDMENT.md`](AMENDMENT.md)

---

## The result

The mechanism E20 tested was the best candidate for E14's collapse: with an
untrained value head and no reward, FPO's advantages are the head's noise,
normalised to full strength, and a precise policy spends its little room on
directions unrelated to the task. E20 gave good HalfCheetah policies exactly
that condition - E16's random critic, and every reward multiplied by zero -
for twenty iterations.

**They did not collapse.** P2 asked for the `zero` arm to fall more than 500
below its start on two of three seeds:

| seed | start | `reward` arm, end − start | `zero` arm, end − start | `zero` − `reward` |
| --- | --- | --- | --- | --- |
| 0 | 1129.8 | +104.7 | **−113.3** | −218.0 |
| 1 | 1632.3 | +382.0 | **+141.2** | −240.8 |
| 2 | 1183.0 | −55.7 | **−353.4** | −297.7 |

**P2 is falsified, 0 of 3** - and by amendment 1, declared before the `zero`
arm had started, that number is *reported, not read as a registered result*,
because P1 failed first (below). A collapse here would have meant a return
back near a random policy's −300, a fall of 1,400 to 1,900. The deepest fall
was 353, and one seed rose.

What the comparison does show, as a description: on all three seeds the
`zero` arm ended **218 to 298 below** the `reward` arm. The spread between two
runs of an identical condition, which this experiment turned out to measure
(next section), was 15 to 85. Removing the reward costs these policies
something real. It does not destroy them.

---

## P1 failed, and what it measured instead

P1 asked the `reward` arm, E16's `except-critic` arm run again, to reproduce
E16 to within 1.0 per seed. It did not:

| seed | E16 | E20 `reward` | difference |
| --- | --- | --- | --- |
| 0 | +73.0 | +104.7 | +31.7 |
| 1 | +366.6 | +382.0 | +15.4 |
| 2 | +28.8 | −55.7 | −84.5 |

The premise was that one-client training is deterministic here, and it was an
overclaim: **collection is deterministic, learning is not**. Two runs of one
configuration agree until the first update and then drift - by 21% after one
update on seed 2 (amendment 1). So the table above is the run-to-run noise of
twenty-iteration gains in this setup, **up to 85**, and it is the scale any
single-run comparison on this line has to clear. E18's first draft and a
comment on #49 had leaned on the same premise; both were corrected.

P3, all six runs under 60 minutes, **holds**: 19.1 minutes.

---

## How far the policies moved

The number that connects E20 to pi0.5 is not the return but the distance.
Relative movement of the actor's weights from the checkpoint each run restored,
[`actor_movement.py`](actor_movement.py), output in
[`results/actor-movement.txt`](results/actor-movement.txt):

| seed | `reward` arm | `zero` arm |
| --- | --- | --- |
| 0 | 0.128 | 0.214 |
| 1 | 0.108 | 0.150 |
| 2 | 0.161 | 0.218 |

Under pure critic noise these actors moved **15% to 22%** of their norm and
kept 70% to 109% of their return. pi0.5's action expert moved **0.66% to
1.5%** under FPO (E15) and kept none of it.

So the mechanism as registered does not carry over on its own. Noise
advantages at full strength move a policy a long way without destroying it,
here. Either pi0.5 has almost no room in any direction - a statement about
the policy, not about FPO - or what FPO does to pi0.5 is not what noise does
to this network: a direction, not a size. Those two are separable without
training anything, and E21 separates them.

---

## What this does and does not support

**Supported:**

* On HalfCheetah, good FPO policies given a randomly initialised value head
  and zero reward for twenty iterations do not collapse. Across three seeds
  they changed by −353 to +141 from starts of 1130 to 1632.
* Removing the reward left every seed 218 to 298 below the same run with it,
  against a run-to-run spread of up to 85 for identical runs. Reported as a
  description, per amendment 1.
* One-client FPO training on this machine is not reproducible run to run past
  the first update; the twenty-iteration gain varies by up to 85 between
  identical runs.

**Not supported:**

* Anything about sparse reward. Zero reward is its limit, not the thing
  itself, though for E14's last eight iterations it is exactly the condition
  (known item 4).
* That noise advantages are harmless to pi0.5. A 4,390-parameter actor
  trained by RL and a 3B-parameter flow VLA trained by imitation need not have
  the same room; this is what E21 measures.
* Why learning is not deterministic. Parallel CPU reductions ordered by
  thread scheduling are the usual cause; it was not isolated.

---

## Reproducing

```bash
bash run.sh                 # six runs, two batches of three; needs E16's Phase A
python summarise.py         # P1, P3, then P2 as amendment 1 reports it
python actor_movement.py    # the distances above
```
