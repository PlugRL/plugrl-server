# E22: FPO's damage to pi0.5 is in its update to the MLP, and in where that update points

2026-09-25 · qz103, 8 × 24 GB GPUs · pi0.5 on LIBERO-10 task 8, fifty
episodes per arm, E14's harness unchanged · protocol:
[`PROTOCOL.md`](PROTOCOL.md)

---

## The result

E21 (#55) showed that FPO's first update destroys pi0.5 by the directions it
moves in, not by how far, and in either sign. It left one reading open. FPO's
update is low-rank, and E21's random arms were not, so the damage could be
in *where* the update points or only in *how concentrated* it is. E22 kept
every matrix's singular values exactly and randomised only the subspace, and
applied FPO's own update to one module group at a time.

| arm | what it is | action expert moved | score of 50 |
| --- | --- | --- | --- |
| *null* | *seven evaluations of an unperturbed actor* | *0* | *28 to 37, 30.9 ± 3.6* |
| `rot-s1` | FPO's spectrum per matrix, random subspace | 1.52% | **28** |
| `rot-s2` | a second subspace | 1.52% | **33** |
| `rot-s3` | a third | 1.52% | **29** |
| `keep-in-s1` | FPO's input side, `rot-s1`'s output side | 1.52% | **31** |
| `only-mod` | FPO's update to the adaRMS modulation only | 0.76% | **25** |
| `only-attn` | to attention only | 0.72% | **26** |
| `only-io` | to the projections and time MLP only | 0 in the expert | **24** |
| `only-mlp` | **to the MLP only** | 1.09% | **0** |
| *E14* | *FPO's whole update* | *1.51%* | *0* |

V1 to V5 pass ([`results/construction.txt`](results/construction.txt)):
the four groups rebuild E14's actor bit for bit, every rotation arm moved
FPO's distance to within 0.4%, and kept FPO's concentration - mean top-8
singular-value share 0.738, 0.558, 0.974 in attention, MLP and modulation,
against FPO's 0.747, 0.565, 0.974. All eight evaluations are valid.

**P1 holds**: an update with exactly FPO's concentration and size, pointed
into a random subspace, scores 28, 33 and 29. **The concentration is not the
cause; the subspace is.** **P2 is falsified**: I predicted the modulation
alone would reproduce the collapse, and it scores 25. **FPO's update to the
MLP alone does**: `only-mlp` scores 0, and like E14's collapsed policy it ran
every one of its fifty episodes to the 520-step limit.

---

## It is not how much each layer's output changes

`keep-in-s1` is the sharpest arm here. It keeps FPO's input side and
randomises only the output side, so for every input a layer actually sees, it
changes that layer's output by **exactly** as much as FPO's update does - only
the direction of the change differs. It scores **31**.

So the damage is not the size of the change FPO makes to each layer's
activations on real inputs. The same change, pushed in other output
directions, is absorbed. What destroys the policy is the particular
directions FPO pushes the MLP's outputs in. That is the reading PROTOCOL.md
declared in advance for a `keep-in` score of 20 or more.

---

## Where it lives: the MLP, alone

The four groups partition FPO's update exactly, and only one of them is
sufficient:

| group alone | tensors | its share of the expert's movement | score | states changed of the 36 the null agrees on |
| --- | --- | --- | --- | --- |
| MLP, `gate up down` | 54 | 1.09% | **0** | 26 - every state the base always wins, lost |
| attention, `q k v o` | 72 | 0.72% | 26 | 17 |
| modulation | 74 | 0.76% | 25 | 12 |
| projections, time MLP | 8 | outside the expert | 24 | 12 |

([`results/episode-patterns.txt`](results/episode-patterns.txt); against the
null's own 0 to 3 changed states.) The MLP carries about half of the expert's
squared movement, and it carries all of the destruction. The other three
groups each cost a little on their own - 24 to 26, below every null
evaluation, with 12 to 17 states changed - and none comes close to zero.

Set beside E21, the contrast is in the direction, not the amount. `only-mlp`
moves the expert 1.09% and scores 0. E21's random direction moved it 15%,
fourteen times as far, and scored 26.

---

## What this does and does not support

**Supported:**

* On LIBERO-10 task 8, an update with the same singular values as FPO's first
  update, matrix by matrix, but in random subspaces, leaves pi0.5's success
  rate within or near its null: 28, 33, 29 of 50.
* A perturbation that changes every layer's output by exactly FPO's amounts
  on real inputs, in random output directions, leaves it at 31 of 50.
* FPO's first update restricted to the action expert's 54 MLP matrices is
  sufficient to take pi0.5 to 0 of 50. Restricted to attention, the
  modulation, or the projections, it is not: 26, 25, 24.

**Not supported:**

* That the MLP is *necessary*. E22 shows MLP-only is sufficient and the other
  groups alone are not. It does not show that FPO's update with the MLP left
  out would be harmless, though the other three groups' scores suggest
  it would be damaged rather than destroyed. That is testable directly.
* Which MLP layers, or why FPO's gradient points the MLP's outputs where it
  does. E22 localises the damage to a module type, not a mechanism.
* Anything beyond task 8 and one evaluation per arm.

---

## What comes next

The localisation turns into a candidate fix that can be tested in training,
not just by perturbation: **run FPO with the action expert's MLP frozen**, and
see whether the first iteration still destroys the policy. If it does not,
the collapse has a located cause and a remedy in the same experiment. The
natural companion is the complement arm this experiment did not run -
FPO's update with only the MLP removed - which answers whether the MLP is
necessary as well as sufficient, without training anything.

---

## Reproducing

On qz103, with E14's base dump and first checkpoint in place:

```bash
setsid nohup bash e22/run.sh > e22/run.out 2>&1 < /dev/null &   # build, check, 2 batches of 4
python e22/episode_patterns.py                                  # the per-state comparison
```

Then, locally, with `results/` copied back: `python summarise.py`.
`groups_probe.py` fixed the four groups before the protocol was written; its
output is `results/groups-probe.txt`.
