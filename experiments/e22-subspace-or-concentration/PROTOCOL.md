# E22 measurement protocol (pre-registered)

**Written 2026-09-25, before any E22 checkpoint was built or evaluated.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E21 (#55) showed that FPO's first update destroys pi0.5 by the directions it
moves in, not by how far. The same per-element movement with random signs
scores 29, 28 and 25 of 50 and survives ten times the distance, while the
update itself and its mirror image both score 0.

But E21 also measured, after its evaluations, that FPO's update is
**low-rank**: effective rank 3.6 in the adaRMS modulation matrices and 13 to
50 elsewhere. Random signs spread the same norm over nearly every direction,
so E21's random arms differ from FPO's in **structure** as well as direction.
Two readings survive:

* **The subspace.** FPO's update lies along directions this policy is sharp
  in, and a different set of directions with the same concentration would be
  harmless.
* **The concentration.** Any update concentrated like FPO's, at this size,
  destroys the policy, whatever its directions.

**E22 separates them by keeping FPO's update's structure exactly - every
matrix's singular values - and randomising only the subspace it lies in.** It
also asks two things the answer leads to: which side of each matrix carries
the damage, and which module group.

### The constructions

For each 2-D tensor FPO changed, `Δ = U S Vᵀ` (thin SVD). For `nn.Linear`,
`y = W x`, so `V` is the input side - the directions the update reads from
the layer's input - and `U` the output side.

* **`rot`: `U' S V'ᵀ`**, with `U'` and `V'` Haar-random orthonormal. The
  spectrum, and so the concentration and the Frobenius norm, are FPO's exactly.
  The subspace is random on both sides.
* **`keep-in`: `U' S Vᵀ`**, with FPO's input side kept and only the output side
  random. For every input `x`, `‖U' S Vᵀ x‖ = ‖Δ x‖`. The perturbation changes
  each layer's output by exactly as much as FPO's update does, on every real
  input, and only the direction of that change is random. `keep-in-s1` uses
  the same `U'` as `rot-s1`, so the two differ in one thing: whose input side
  they have.
* **1-D tensors**, the 41 biases, get E21's `r ⊙ Δ` in every `rot` and
  `keep-in` arm, since a vector has no spectrum to keep.

And the localisation arms, **`only-<group>`**, apply FPO's own `Δ`, unchanged,
to one module group and leave everything else at the base. The groups,
counted by `groups_probe.py` before this file was written
(`results/groups-probe.txt`), partition the 208 changed tensors exactly:

| group | tensors | what they are |
| --- | --- | --- |
| `mod` | 74, float32 | the adaRMS modulation: every `*_layernorm.dense` and `model.norm.dense`, weight and bias |
| `attn` | 72, bfloat16 | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `mlp` | 54, bfloat16 | `gate_proj`, `up_proj`, `down_proj` |
| `io` | 8, float32 | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out` |

### What this cannot settle

* One task, fifty initial states, one evaluation per arm, as E21.
* `rot` randomises the subspace against *all* structure. If it is harmless,
  E22 shows that FPO's subspace is special. It does not say what makes it
  special - the demonstrations' gradients, or the activations every gradient
  step is built from. `keep-in` is the first cut at that.
* The localisation arms apply one group's share of an update that was
  computed for all four groups at once. A group that is harmless alone may
  still be necessary in combination.

---

## Declared in advance: what was already known

1. **The null**, from E15 correction 2: seven evaluations of an unperturbed
   actor, 28 to 37, **30.9 ± 3.6**. E21's three random-sign arms at FPO's size
   scored 29, 28, 25.
2. **FPO's update and its mirror** score 0 and 0 (E14, E21). E21's random-sign
   arms survive 10× FPO's size (26) and fail at 30× (1).
3. **The spectrum of `Δ`**, from E21's `delta_structure.py`: mean top-8
   singular-value share 0.974 in the 36 layer `dense` matrices, 0.69 to 0.81
   in attention, 0.55 to 0.58 in the MLP.
4. **E15's drift stop measured the wrong thing for this failure.**
   `max_policy_drift` stops on `|1 − mean policy ratio|`, FPO's ratio of
   flow-matching losses on the collected actions. It held that near 1 on
   pi0.5 while the policy was destroyed (E15, both iterations 0 of 50).
   Recorded because it bears on what a fix would have to measure, not on any
   prediction here.

---

## Design

Nine checkpoints are built, eight evaluated:

| arm | tensors changed to | what it asks |
| --- | --- | --- |
| `rot-s1` | `U'₁ S V'₁ᵀ`; 1-D `r₁ ⊙ Δ` | FPO's concentration, a random subspace |
| `rot-s2` | seed 2 | the same, a second subspace |
| `rot-s3` | seed 3 | a third |
| `keep-in-s1` | `U'₁ S Vᵀ`; 1-D `r₁ ⊙ Δ` | FPO's input side, `rot-s1`'s output side |
| `only-mod` | `θ + Δ` on `mod`, base elsewhere | is the modulation enough? |
| `only-attn` | on `attn` | attention? |
| `only-mlp` | on `mlp` | the MLP? |
| `only-io` | on `io` | the projections and time MLP? |
| `fpo` | `θ + Δ` on all four | the check, not evaluated |

`U'` and `V'` come from the QR decomposition of a Gaussian matrix, signs
corrected so the result is Haar-distributed. They are drawn per tensor, in
sorted key order, from a `torch.Generator` seeded with the arm's seed, `U'`
before `V'`. The 1-D signs `r` come from a second generator with the same
seed. All arithmetic is float64, as E21's amendment 1 requires, and each
result is cast once to its stored dtype.

Evaluation is E14's harness unchanged, as in E21: task 8, fifty episodes,
initial states 0 to 49, `runner.seed` 7, two batches of four on card pairs.

* batch 1: `rot-s1`, `rot-s2`, `keep-in-s1`, `only-mod`
* batch 2: `rot-s3`, `only-attn`, `only-mlp`, `only-io`

---

## Checks before any evaluation

`make_arms.py` refuses to exit 0 unless:

* **V1** - `fpo`, built as the union of the four groups, is bit-identical to
  E14's first checkpoint on all 208 tensors, and the four groups are disjoint
  and cover them.
* **V2** - every arm is the base's, bit for bit, on every tensor it does not
  change, read back from the saved file with the same keys, shapes and dtypes.
* **V3** - every `rot` and `keep-in` arm's realised action-expert distance is
  within **5%** of FPO's 0.0151. Every `only-<group>` arm's changed tensors
  are bit-identical to E14's on that group.
* **V4** - the structure survives the cast. In every `rot` and `keep-in` arm,
  the mean top-8 singular-value share of the realised perturbation, per group
  of 2-D matrices (`mod`, `attn`, `mlp`), is within **0.05** of FPO's.

After evaluation, **V5**: every evaluation valid - fifty episodes, client exit
0, `valid` true in the harness's own table.

---

## Predictions, and what falsifies each

Thresholds are E21's, which were checked against the null: an unperturbed
actor has never scored below 28, and under a normal approximation to the null
would score 19 or less about once in a thousand evaluations.

**P1 - it is the subspace, not the concentration.** Each of `rot-s1`,
`rot-s2` and `rot-s3` scores **at least 20 of 50**.

> Holds: FPO's update is destructive because of where it points. An update of
> exactly its concentration and size in a random subspace is not. All three
> at 5 or less: the concentration is enough on its own. Any update this
> concentrated at this size destroys the policy, and the fix has to limit
> concentration, not direction. Anything else: partial, reported, no reading.
>
> Why this prediction: a gradient step's input side is built from the layer's
> own activations, which occupy few directions. A random input side mostly
> misses them, so a random-subspace update of the same norm should move the
> layer's output far less on real inputs.

**P2 - the modulation carries the damage.** `only-mod` scores **at most 5 of
50**.

> The adaRMS modulation conditions every layer of the action expert on the
> flow time, and FPO's update to it is the most concentrated of any group
> (known item 3). Holds: FPO's change to the modulation alone reproduces the
> collapse. Falsified by `only-mod` scoring 6 or more.

**Not predicted, reading declared:**

* **`keep-in-s1`.** At 10 or less: moving each layer's output by FPO's
  amounts, on the inputs the network actually sees, destroys the policy
  whatever direction those output changes take. The damage is in how much
  FPO's update changes what each layer computes, which weight-space distance
  does not measure. At 20 or more: the output directions FPO chose matter
  too, and a change of the same size in other directions is absorbed. 11 to
  19: no reading. Read only if P1 holds. If P1 fails, `rot` already destroys
  and `keep-in` cannot separate anything.
* **`only-attn`, `only-mlp`, `only-io`.** Each reported. At 5 or less, that
  group is sufficient on its own. At 20 or more, it is not sufficient alone.
  If no single group scores 5 or less, the damage needs groups in
  combination.

---

## Declared deviations allowed in advance

1. One restart of any evaluation that dies for a reason outside the
   experiment, recorded in `AMENDMENT.md`.
2. An arm may move to another pair of cards if its own pair is occupied.
3. If a check fails, nothing is evaluated; the construction is fixed and the
   fix recorded in `AMENDMENT.md`, as E21 did.

---

## Reading order

V1 to V4, then V5, then P1, then `keep-in-s1` if P1 holds, then P2, then the
other localisation arms.
