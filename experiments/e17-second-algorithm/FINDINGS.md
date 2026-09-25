# E17: a second algorithm runs, and does not learn in the setting I gave it

2026-09-24 · Windows 11, CPU only, no GPU · `fpo-policy × dppo(cheetah) ×
HalfCheetah-v5`, 272,423 parameters · protocol:
[`PROTOCOL.md`](PROTOCOL.md)

---

## The result, in the order the protocol fixes

**P4 — wall clock. Holds.** Three concurrent seeds, 409,600 steps each, from
22:21:14 to 23:32:18: **71 minutes** against a 150-minute bound. All three
servers stopped on their own signal, no traceback in any log.

**P3 — the critic learns. Falsified.** `rollout/explained_variance` ends at
0.428, 0.379 and 0.291, against a bar of 0.5.

**P1 — DPPO learns. Falsified.** The bar was `end` exceeding `start` by more
than 500. Measured, on the protocol's mean-of-ten:

| seed | start | best | end | end − start |
| --- | --- | --- | --- | --- |
| 0 | −291.1 | −281.1 | −312.3 | **−21.2** |
| 1 | −313.1 | −281.1 | −321.5 | **−8.4** |
| 2 | −289.6 | −266.5 | −318.1 | **−28.5** |

All three rise slightly and then fall back below where they started. FPO on
this same line, in E16's Phase A, goes from −328 to +1706.

**P2 — it holds what it learns. Holds, and means nothing.** On the shifted
scale the ratios are 1.36, 1.51 and 1.63, all inside the factor of two. A
policy that learned almost nothing cannot lose much of it, and the protocol's
own reading order puts P2 last for this reason.

---

## What is actually supported

**A second algorithm runs end to end through PlugRL.** `dppo` was on the CLI
menu for the life of the project and had never executed — selecting it died
with `KeyError: 'Algorithm dppo is not registered.'` It now trains a policy
across the boundary for 409,600 steps, three seeds, on a laptop CPU with no
extras installed, and stops cleanly. That is the platform claim E17 was run to
test, and it holds.

**The machinery works.** The critic goes from `explained_variance` −0.0004 to
0.428 — it learns to predict returns from nothing. The actor moves too, and
substantially: between the 81,920 and 409,600 checkpoints its ten tensors are
**5.15e-02 apart, relative, with none identical** (the critic moves 3.92e-01;
`obs_stats_*` are bit-identical, as they should be for a policy whose
normalisation is frozen after warmup).

**And the policy does not get better.** That is the whole of it: gradients
flow, weights move five per cent, and the return does not improve.

---

## This is the configuration, not the algorithm

Per the project's standing rule, a negative from a platform is a defect in how
the platform was used until shown otherwise. `PROTOCOL.md` registered no
prediction about which algorithm scores higher, for exactly this reason, and
none should be read out of the table above.

**The design error is mine and it is not subtle. DPPO is a fine-tuning
method.** Its paper pretrains a diffusion policy by behaviour cloning and then
does RL on it; that is the setting the algorithm exists for. E17's "held
fixed" section matched E16's Phase A down to the seed, and in matching it put
DPPO in front of a **randomly initialised policy**, which is not a setting it
was designed for. The protocol says the two arms share "the policy, the
environment, the buffer size and the step budget" and treats that as
conservative. It is not: for one of the two algorithms it changes the problem.

Two more configuration facts, neither established as the cause:

* **Its buffer was overridden from 20,000 to 4,096** to match the FPO arm.
  With `batch_size` 2048 that leaves **two minibatches per epoch**, which is
  not a regime anyone tuned the cheetah variant for.
* **`grad_accum_steps` defaults to 8 and the harness did not set it.** The
  update loop divides each minibatch's loss by that 8 while only two
  minibatches accumulate before the end-of-epoch forced step, so the applied
  gradient is **a quarter of the correct scale**. That is a defect in
  `learn_impl` independent of this experiment and is worth fixing on its own;
  a factor of four does not explain a flat curve, but it should not be there.

---

## What would settle it, and the gap in the way

The experiment DPPO deserves is the one it was built for: start from a policy
that is already good and fine-tune. E16 leaves exactly that lying around —
three checkpoints at step 327,680 scoring 1129.8, 1632.3 and 1183.0 — and the
run would be twenty updates, about fifteen minutes.

**It cannot be run yet.** `policy_checkpoint_path` and `--algo.restore` were
added to `FPOAlgoConfig` in PR #39 and not to `DPPOAlgoConfig`, so DPPO has
the same gap FPO had until this week: a `load_checkpoint` nothing can reach.
Closing it is small, and it is the next thing to do here.

---

## A pattern in my own predictions, worth naming

Two experiments, two thresholds that did not mean what I wanted them to.

E16's P2 compared one seed against the spread across three, so the best seed
failed it by being itself. E17's P3 set `explained_variance > 0.5` and called
anything below it "a value head that cannot predict returns", which would make
every advantage noise — and then measured 0.29 to 0.43, a critic that plainly
learned a great deal and simply did not reach an arbitrary bar. Both
predictions are falsified as written, both were reported as falsified, and in
both cases the falsification does not mean what the protocol said it would.

The fix is not to loosen thresholds afterwards. It is to check, before
registering a number, what that number takes under the null and under the
outcome being predicted. Neither of these was checked.

---

## What this does and does not support

**Supported:**

* PlugRL trains a policy with a second algorithm, end to end, with no extras
  installed: 100 iterations × 3 seeds × 409,600 steps, clean exits.
* DPPO's critic learns on this task: `explained_variance` −0.0004 → 0.428.
* Its actor moves: 5.15e-02 relative over 80 iterations, no tensor unchanged.
* 71 minutes for three concurrent seeds on a laptop CPU.

**Not supported:**

* Anything about DPPO as an algorithm. It was run from a random initialisation
  when it is a fine-tuning method, on a buffer a quarter of its configured
  size, with a gradient mis-scaled by four. Any of those is enough to make the
  curve uninformative about DPPO.
* Any comparison of FPO against DPPO. The protocol registered no such
  prediction and this document makes no such claim.
* That the actor's movement was *useful* movement. Five per cent of relative
  change with no return improvement is consistent with a gradient pointed
  somewhere unhelpful, and E17 does not say where.

---

## Reproducing

```bash
bash run.sh                  # 3 seeds x 409,600 steps, ~71 min, three at a time
```

Curves are in each seed's tensorboard under `results/dppo/fpo-policy/`; the
checkpoints and event files stay out of the repository and the logs are
committed whole.
