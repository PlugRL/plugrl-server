# E25: pi0.5 trains end to end under DPPO, and two iterations leave it intact - because they barely move it

2026-09-25/26 · qz103 · `pi0-policy` `pi05_libero` · DPPO's `libero` variant ·
LIBERO-10 task 8, two iterations of 4,096 · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (after the pilots in
[`results/pilot.txt`](results/pilot.txt)) · code `main` + #58

---

## The result

Until E25 the platform had trained pi0.5 with one algorithm, FPO, whose
first iteration takes it from about 29 of 50 to 0 (E14, E26). DPPO has a
`libero` variant written for this setting that had never run.

**It runs, and its two iterations do not destroy the policy.**

* **P1 holds.** Two iterations logged, a checkpoint after each (4,100 and
  8,200 steps), no traceback, the server exited on its own after 7,683 s,
  and card 0 peaked at 21,318 MiB of its 24,576.
* **P2 holds.** The last checkpoint scores **30 of 50** (0.60, 95% CI
  0.46-0.72), inside the null of 28-37 and far from FPO's 0. The evaluation
  is valid: fifty episodes, client exit 0, 22,959 steps reconciled between
  client and server.

**But DPPO hardly moved it.** Relative distance from the base after DPPO's
two iterations, beside FPO's one (E14's checkpoint, the same measurement):

| group | DPPO, 2 iterations | FPO, 1 iteration | DPPO / FPO |
| --- | --- | --- | --- |
| modulation (74 tensors) | 0.258% | 2.111% | 0.12 |
| attention (72) | 0.031% | 1.663% | 0.019 |
| MLP (54) | 0.031% | 1.551% | 0.020 |
| projections (8) | 0.250% | 1.320% | 0.19 |

The MLP, where E22 found FPO's damage, moved a fiftieth as far. In
training, `approx_kl` was 0.0000 to four decimals in both iterations and
`clipfrac` 0.085 and 0.072.

So P2 holds for the reason the protocol gave - DPPO's updates are too small
to do FPO's damage in two iterations - and that is a weak result, the same
one E19 found on HalfCheetah (DPPO moved a trained actor 0.28-0.43% where FPO
moved it 11-16%). It does not say DPPO's *direction* is harmless: E21 showed
that the size of an update is not what matters, and nothing here moved far
enough to test the direction.

---

## What it took to run

The first DPPO training of a VLA on this platform found one defect and one
limit before the registered run (`results/pilot.txt`):

1. **A defect, fixed in #58.** pi0's per-camera `image_mask` is a per-sample
   scalar; DPPO's DataLoader indexes the buffer by integer, which returned a
   numpy scalar that the collate step took for a mapping. The server died on
   its first learning step.
2. **A limit.** The variant's minibatch of 128 runs out of memory in pi0.5's
   VLM prefix on a 24 GB card, and DPPO has no option, as FPO does, to keep
   its optimiser's copies on a second card - card 1 sat at 5 MiB. The run
   used minibatch 8 with the variant's 16 accumulation steps, so each
   optimiser step still saw 128 samples.

---

## What this does and does not support

**Supported:**

* pi0.5 trains end to end under DPPO on PlugRL - the second algorithm to
  train a VLA here - on one 24 GB card at minibatch 8.
* Two DPPO iterations leave its LIBERO task-8 success where it was, 30 of 50.

**Not supported:**

* That DPPO improves pi0.5. The training rollouts' mean return - 1 for a
  success - went from 0.65 to 0.68, which two iterations cannot distinguish
  from noise; nothing about improvement was registered.
* That DPPO's update direction is safe for pi0.5 at a size that would matter.
* Anything beyond two iterations, this task or this seed.

**A caveat on the evidence of code**, as in E26. The evaluation harness
hashed the server's source without `PYTHONPATH`: its manifest records
`buffer/numpy_tree_storage.py` as it was before #58
(`d9f5c71e...`, 8812b54's), not the fixed file that `$R/plugrl-server-main`
holds. What shows the fixed code ran is the training itself: without #58
the first learning step crashes (pilot 1), and here both completed. The
evaluation path does not touch the buffer.

---

## Reproducing

On the cluster, with this directory copied to `$R/e25` and `main` + #58
deployed LF at `$R/plugrl-server-main`: `setsid nohup bash run.sh > run.out
2>&1 &` - about three hours: two iterations of training, the movement, then
the evaluation. Then, from `$R/e25` (it needs `tensorboard`):

```bash
python summarise.py   # P1, P2, the training figures and the movements, from results/
```

`results/` holds the cell's output, the server's and clients' logs, the card
memory samples, the tensorboard file, the harness's tables and both movement
measurements. The checkpoints stay on the cluster.
