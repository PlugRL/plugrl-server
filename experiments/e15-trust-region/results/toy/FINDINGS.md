# The toy: FPO learns a bandit and does not hold it

Ten hypotheses about E14's collapse have been refuted by experiment. Five of
them cost hours each on the cluster, one knob at a time, and none of them
could have found a defect, because defects do not live in hyperparameters.
The control that was missing all along is here: a bandit on a two-layer flow
policy, CPU, about a minute per run, where the right answer is known.

Every other FPO test in this repository feeds `reward=1.0`, which cannot tell
learning from drift. So until this, nothing had shown FPO makes a policy
better at anything, or keeps it better.

## What is established

**FPO learns.** From about -1.5 to about -0.06 within twenty iterations, on
every seed and in both precisions.

**FPO does not hold what it learns.** Final reward against the best reached,
thirty iterations, five seeds. 1.0 means the policy kept what it found:

| | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| float32 | 1.05 | 1.01 | **2.72** | 1.70 | **4.59** |
| bfloat16 | **9.00** | 1.17 | 1.82 | 1.50 | **2.76** |

Four of ten fail a loose two-times bar. `tests/test_fpo_learns_anything.py`
is that table as an executable test, marked `xfail(strict=False)` - four of
its ten cases fail, and a suite that is red on purpose cannot report anything
else on a repository whose CI runs pytest on every pull request.

**There is no dtype effect.** An earlier version of this work claimed the
collapse was specific to bfloat16 and localised it to the master-weights
path, on the strength of the 9.00 above - one seed. float32 ends 4.59x worse
on seed 4 and 2.72x on seed 2, and on seeds 2, 3 and 4 bfloat16 held better.
The claim was withdrawn.

**A diverging value loss tracks it, closely.** Reward's fall from its best
against value loss's rise from its lowest, same ten runs:

```
            reward fall   vloss rise
bfloat16 0      9.00         55.53
bfloat16 1      1.17          1.16
bfloat16 2      1.82          2.68
bfloat16 3      1.50          1.90
bfloat16 4      2.76          6.29
float32  0      1.05          1.34
float32  1      1.01          1.00
float32  2      2.72          3.64
float32  3      1.70          2.33
float32  4      4.59         12.48
```

Agreement on a two-times bar: 8 of 10. Rank correlation: **0.988**. The
association held again across fifteen further runs at three different value
loss coefficients.

Over the same stretch the flow-matching loss keeps falling and the policy
ratio stays near 0.99 in both the runs that hold and the runs that do not.
The critic's behaviour is the only thing that separates them.

## What is not established

**The direction.** `value_loss_coeff` scales how hard the critic is pushed,
and damping it does not break the coupling - the worst fall across five seeds
goes 4.59x at 0.25, 4.14x at 0.05, 3.79x at 0.01, and per seed the effect is
mixed rather than monotone. That points away from the critic as the cause and
towards it as a symptom: the policy drifts, the returns stop being
stationary, and the value loss rises because it is chasing a moving target.

**That this is E14's mechanism.** E14's `losses/value_loss` fell
monotonically, 0.880 to 9.3e-05, because once no episode earned a reward
there was nothing left to predict. Its policy died in one iteration, before a
critic could diverge. The toy degrades slowly over twenty. They share a shape
- learn, then lose it - and that is all that has been shown.

## What it is good for

A detector. The practical problem in E14 was that training destroyed the
policy and it took a forty-five minute evaluation to find out. A quantity
already logged every iteration, which ranks with the damage at 0.988, is an
early stop that needs no evaluation at all.

## Refuted along the way

| hypothesis | how |
|---|---|
| advantage normalised per minibatch rather than per buffer | two iterations on pi0.5, 0 of 50 twice |
| too many gradient steps | `num_updates_per_batch` 4 to 1, 0 of 50 |
| learning rate too large | 1e-5 to 1e-6, 0 of 50 |
| trust region too wide | five-point sweep 0.05 to 0.002, not monotone |
| critic starts uninformative | one warmup iteration, actor frozen, then 0 of 50 |
| bfloat16 and the master weights | five seeds, float32 fails too |
| unbounded weight growth | weight decay, helped one seed of three, hurt another |
| advantage normalisation itself | turning it off stops learning and blew one run to -9.3e6 |
| the critic being pushed too hard | `value_loss_coeff` 0.25 to 0.01, collapse persists |
| the policy drifting too far within an iteration | `max_policy_drift` 0.01 - the best of five settings on this bandit, worst fall 9.00x to 2.14x - gave 0 of 50 on both pi0.5 iterations |

Each of these was a real possibility when it was proposed and each was
settled by a measurement rather than an argument. The scripts that produced
every number are in this directory.
