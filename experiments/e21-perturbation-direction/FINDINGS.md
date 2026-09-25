# E21: FPO destroys pi0.5 by the directions it moves in, not by how far

2026-09-25 · qz103, 8 × 24 GB GPUs · pi0.5 on LIBERO-10 task 8, fifty
episodes per arm, E14's harness unchanged · protocol:
[`PROTOCOL.md`](PROTOCOL.md) · amendment: [`AMENDMENT.md`](AMENDMENT.md) - the
construction check failed once and was fixed before anything was evaluated

---

## The result

One FPO iteration takes pi0.5 from about 30 of 50 to 0 (E14), after moving its
action expert **1.5%** of its norm (E15). Two explanations were left. Either
pi0.5 has no room for a move of that size in any direction, or the direction
FPO chose is what destroys it. E21 built checkpoints that move pi0.5 exactly as
far as FPO's first update did, and evaluated them without training anything.

* **Random:** `θ + k · r ⊙ Δ`. Every element moves by exactly what FPO moved
  it, and only the signs `r` are random.
* **Mirror:** `θ − k · Δ`, FPO's own update run backwards.

| arm | what it is | action expert moved | score of 50 |
| --- | --- | --- | --- |
| *null* | *seven evaluations of an unperturbed actor* | *0* | *28 to 37, 30.9 ± 3.6* |
| `flip-s1` | random signs, FPO's size | 1.51% | **29** |
| `flip-s2` | a second sign pattern | 1.51% | **28** |
| `flip-s3` | a third | 1.51% | **25** |
| `flip-s1-x3` | `flip-s1`'s direction, 3× as far | 4.53% | **30** |
| `flip-s1-x10` | 10× | 15.1% | **26** |
| `flip-s1-x30` | 30× | 45.3% | **1** |
| `neg` | FPO's update, backwards | 1.51% | **0** |
| `neg-x3` | backwards, 3× as far | 4.53% | **0** |
| *E14* | *FPO's update itself* | *1.51%* | *0* |

**P1 holds**: all three random directions at FPO's size score at least 20 -
29, 28, 25. **P2 holds**: FPO's update run backwards scores 0, against a bound
of 10. V1 to V4 pass. Every checkpoint changes only the 208 tensors FPO
changed, each at its registered distance to within 0.1%. All eight evaluations
are valid, and all eight ran the same source, one manifest hash.

**Movement of this size is harmless. FPO's direction is not, in either
sign.** In a direction FPO did not choose, pi0.5 keeps its skill at ten times
the distance - 15% of the action expert's norm - and loses it only at thirty.
Along FPO's own direction, forwards or backwards, 1.5% takes it to zero, and
`neg` and `neg-x3` ran every one of their fifty episodes to the 520-step limit,
as E14's collapsed policy did. Per unit of movement, FPO's direction is at
least ten times as destructive as a random one.

So E15's lesson - that under 1% of movement is enough to destroy this policy -
was true only of FPO's direction. It is not a property of pi0.5.

---

## The random arms really are different policies

A random perturbation that did nothing would also score 29. It did not do
nothing ([`episode_patterns.py`](episode_patterns.py), output in
[`results/episode-patterns.txt`](results/episode-patterns.txt)). The seven null
evaluations agree unanimously on 36 of the 50 initial states: 26 always
succeed, 10 always fail. Measured against the other six, each null evaluation
departs from that consensus on **0 to 3** states. The random arms depart from
it on **5 to 13**:

| arm | score | states changed of the 36 | lost / won |
| --- | --- | --- | --- |
| `flip-s1` | 29 | 11 | 7 / 4 |
| `flip-s2` | 28 | 5 | 4 / 1 |
| `flip-s3` | 25 | 9 | 8 / 1 |
| `flip-s1-x3` | 30 | 9 | 6 / 3 |
| `flip-s1-x10` | 26 | 13 | 10 / 3 |
| `neg`, `neg-x3`, `flip-s1-x30` | 0, 0, 1 | 26 | every always-won state lost |

A random move of FPO's size changes which states the policy solves, and keeps
how many it solves. It loses states the base always wins and wins states it
always loses, in roughly the proportion the totals need.

One thing in this table leans the other way, and is reported rather than
explained away. The three random arms at FPO's size average 27.3, below the
null's 30.9, and `flip-s3`'s 25 is below every null evaluation. So a random
move of this size may cost a little. It is nothing like the destruction:
"a little" is three states of fifty, and FPO's direction costs all of them.

---

## What "either sign" does and does not establish

P2 was registered to split the direction explanation in two. First order: the
update points the wrong way, as a sign or credit-assignment error would make
it. Second order: FPO moves along directions the pretrained policy is sharp
in, and either sign hurts. **The mirror destroys the policy as completely as
the update does, so it is not a sign error.** That reading is registered and
it holds.

What the mirror cannot say is *why* those directions are sharp, and a
measurement taken after the evaluations shows why that matters
([`delta_structure.py`](delta_structure.py), output in
[`results/delta-structure.txt`](results/delta-structure.txt)). FPO's update is
**low-rank**. Across the action expert's 167 weight matrices:

| matrices | FPO's `Δ`, effective rank | top singular value's share | after random signs, effective rank |
| --- | --- | --- | --- |
| adaRMS modulation (`dense`, 36) | **3.6** | 66% | 713 |
| attention (`q k v o`, 72) | 13 to 23 | 33% to 42% | 200 to 724 |
| MLP (`gate up down`, 54) | 40 to 50 | 24% to 27% | 826 to 862 |

A gradient step is a sum of outer products over a batch, so this is expected,
and randomising each element's sign spreads the same norm over nearly every
direction. **The random arms differ from FPO's in structure as well as in
direction.** Two readings remain, and E21 cannot separate them:

* **FPO's subspace is the sharp one.** The directions its gradients point
  along are the ones the demonstrations pinned down. With advantages that are
  almost all an untrained value head's noise (E20, known item 4), the update is
  a random combination of per-sample flow-matching gradients, and it lands in
  exactly that subspace.
* **Any concentrated move of this size is harmful.** A low-rank change moves a
  layer's output far more than an unstructured change of the same norm. If
  that is the cause, any update concentrated like FPO's would do the same
  damage, whatever its directions.

The first puts the fault in what FPO's advantages carry. The second puts it in
how an update of this shape meets this network. They lead to different fixes,
and E22 is designed to tell them apart.

---

## What this does and does not support

**Supported:**

* On LIBERO-10 task 8, pi0.5's action expert tolerates a movement of 15% of
  its norm in a random direction built from FPO's own per-element magnitudes
  - 26 of 50 against a null of 28 to 37 - and fails at 45%, 1 of 50.
* The same movement along FPO's first-iteration update, at 1.5%, scores 0 in
  either sign, and so does the mirror at 4.5%.
* FPO's destruction of pi0.5 is not a sign or credit-assignment error that
  points the update the wrong way. Reversing the update destroys the policy
  as completely.
* FPO's first-iteration update is concentrated in a few directions per matrix:
  an effective rank of 3.6 in the adaRMS modulation matrices and at most about
  fifty anywhere in the action expert.

**Not supported:**

* That FPO's specific subspace is what is sharp, as opposed to any update of
  its rank and size. That is E22's question.
* Anything beyond task 8 and its fifty initial states, or beyond one
  evaluation per arm. The null was measured on the same task and states.
* That random moves at FPO's size are exactly harmless. They average 27.3
  against 30.9, and one scored below every null evaluation.

---

## Reproducing

On qz103, with E14's base dump and first checkpoint in place:

```bash
setsid nohup bash e21/run.sh > e21/run.out 2>&1 < /dev/null &   # build, check, 2 batches of 4
python e21/episode_patterns.py                                  # the per-state comparison
python e21/delta_structure.py BASE FPO                          # the rank structure of Δ
```

Then, locally, with `results/` copied back:

```bash
python summarise.py    # V1-V4, P1, P2, P3 in the protocol's order
```

`probe_delta.py` and `diagnose_v1.py` are the two probes that shaped the
construction; their outputs are in `results/`. The nine checkpoints, 8.5 GB
each, are not committed. `make_arms.py` rebuilds them bit for bit from the
two inputs, and `results/arms.sha256` holds their hashes.
