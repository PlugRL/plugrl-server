# E35: fpo-policy cloned on DPPO's square demonstrations succeeds 0.44 of the time

2026-09-27 · clone on qz103 (one RTX 3090), evaluation on `guangzhao` (CPU)

---

## What this is

Not a test of a prediction: E35 makes the starting policy the
`fpo-policy` square cells need. Both algorithms' authors fine-tune square
from a policy trained on demonstrations - DPPO from its released diffusion
policy, FPO++ from behaviour-cloned flow policies - and `fpo-policy` had none.
E27 had started it from random weights, where square's sparse reward gives
nothing to learn from.

`pretrain.py` clones `fpo-policy` on the 300 multi-human demonstrations DPPO
pretrained its own square policy on (`train.npz` as DPPO processed it, 80,731
steps, 79,831 chunks of 4), taken back to the environment's units with DPPO's
`normalization.npz`: chunks of 4 actions, three hidden layers of 1024 (the
width and depth of DPPO's square network), the flow-matching loss FPO++
scores with (velocity error, one uniform time per sample), the target the
chunk through atanh (the latent `fpo-policy`'s tanh maps onto the action),
AdamW at 1e-4 with cosine decay, 400,000 steps of 256, an exponential moving
average of the weights saved. The observation statistics are the data's.

## The clone

| | |
| --- | --- |
| steps | 400,000 of 256, 17.5 minutes on one card |
| loss | 0.515 at 5,000 steps, 0.150 at the end |
| elements clipped to ±0.999 | 14.8% - the gripper, which the demonstrations drive to ±1 |
| checkpoint | `model.safetensors`, 9,838,028 bytes, sha256 7fe7a706c2c4601d |

## How often it succeeds

Fifty episodes each with the `eval` algorithm (the flow's ODE, no sampling
noise), `robomimic-v1` on `square-img`, episodes of at most 400 steps ending
on success, replanning every 4 steps:

| flow steps | successes of 50 |
| --- | --- |
| 10 (FPO++'s) | **22** (0.44) |
| 20 (DPPO's, and E33's for `fpo-policy` under DPPO) | **19** (0.38) |

DPPO's released square policy starts at about 0.40 in its paper's
deterministic evaluations. The clone is a comparable start.

## Files

* `pretrain.py`, `eval_bc.sh` - the clone and the evaluation.
* `results/pretrain.json`, `results/pretrain.out` - settings and loss curve.
* `results/eval-flow{10,20}.out`, `results/eval-flow{10,20}/` - each
  evaluation's log, the client's summary and the server log.

The checkpoint stays on `guangzhao` (`~/zuogou/plugrl/ckpt/e35-bc/400000/`)
and the cluster (`/home/gotham/tmp/plugrl/e35/bc/400000/`).
