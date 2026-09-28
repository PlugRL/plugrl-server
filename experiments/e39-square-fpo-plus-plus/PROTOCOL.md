# E39 measurement protocol (pre-registered)

**Written 2026-09-27, after the pilot in `results/pilot.txt` and before the
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E37 fine-tuned E35's behaviour-cloned `fpo-policy` on robomimic square
under FPO with FPO++'s policy loss. It fell: 0.50 to 0.24-0.42 over fifty
evaluation episodes. DPPO lifted the same clone to 0.78-0.90. E37's critic
never fit the returns. By this project's rule that is our defect. And E37 ran
only FPO++'s loss, not the rest of its fine-tuning: its critic learned at the
actor's rate on rewards scaled by ten, among other differences. #91 adds the
rest as options.

**Run with FPO++'s square fine-tuning in full, as far as a state-based clone
with chunks of 4 allows, does `fpo-policy` learn square under FPO?**

### What this cannot settle

* Which of the added settings mattered. They are added together.
* Comparison with FPO++'s numbers. Its policy reads images, acts in chunks of
  16, and was cloned on other demonstrations.

---

## Declared in advance: what was already known

1. **E37**, same clone, FPO++'s loss only, 60 iterations. Training success
   went from 0.52-0.55 to 0.26-0.39. Fifty-episode evaluations: 0.50 for the
   clone, 0.24 / 0.30 / 0.42 for the three seeds' final checkpoints. Under
   DPPO it went to 0.78 / 0.86 / 0.90. Value loss went from about 9 x 10⁴ to
   about 4 x 10⁴, with 42-50% of samples clipped over the last ten
   iterations.
2. **FPO++'s own square runs** (arXiv 2602.02481 Fig. 4, read off the plot,
   approximate). About 0.28 at the start and about 0.40 at the first
   evaluation. At 8M steps, about 0.58 with zero-sampling and about 0.54 with
   random sampling. No FPO++ curve falls. An adapted pipeline with chunks of 4
   (App. D.3) reaches about 0.9 by about 9.5M steps.
3. **The pilot** (`results/pilot.txt`): seed 0, three iterations.
   - Success: 0.45, 0.55, 0.53.
   - Explained variance: −0.06 (critic-only), then 0.23, 0.33.
   - Clipped fraction: 21%.
   - About 250 s an iteration.
4. **E37 ran with the GAE episode-boundary defect** (#86). E39 runs without
   it.

---

## Design

One cell, `fpopp-square`, on `guangzhao`. Three seeds, each its own server
and three client processes of `robomimic-v1` on `square-img` (84 x 84
agentview, read by nothing). Episodes last at most 400 steps and end on
success, as FPO++'s do. The client replans every 4 steps. **100
iterations** of 12,000 chunks (48,000 steps), 4.8M steps in all.

The start is E35's clone, restored with `except-critic`, with its
observation statistics frozen (#82).

The settings are E37's policy loss plus #91's options, as `run_cell.sh`
passes them.
- Policy loss:
  - chunk loss over the 4 executed steps and 7 dimensions, summed over steps
  - velocity error, Huber δ = 1
  - uniform flow times
  - one ratio per sample, clip 0.01
  - 8 samples per action, 10 flow steps
- Optimizer:
  - AdamW, actor at 1e-5 with betas (0.9, 0.99)
  - critic at 1e-4
  - eps 1e-5, weight decay 1e-6
  - each clipped to a gradient norm of 25
- Advantages:
  - normalised per minibatch
  - computed once per iteration (`--algo.no-fpo-playground-trick`)
- Rewards and value loss:
  - raw rewards
  - a truncated episode treated as ended
  - value loss 0.5 x squared error
- Schedule:
  - one critic-only iteration
  - 10 epochs of 8 minibatches
  - gamma 0.995, lambda 0.99
- Value head: two layers, 512 and 256.

Not FPO++'s:
- The value head uses SiLU, not ReLU.
- Its input is the normalised state, not image features.
- The chunk is 4, not 16.
- There are 3 client processes, not 30 environments.
- The environments are not reset at every iteration.
- The per-sample log-ratio is clamped straight-through at ±5, where FPO++'s
  square run has no clamp.

Code: this branch (#91 plus this directory). Client: plugrl-env-client at
931ab56.

---

## Checks

* **V1 - the settings took**: every seed's server log restores the clone with
  `except-critic` and shows every setting `summarise.py` lists. A seed that
  fails is not read.
* **V2 - the critic learns** (reported; it does not decide the status): mean
  explained variance over iterations 11-20 above 0.2 on every seed.

---

## The status rule

E37's. A seed's start is its mean `rollout/success` over iterations 1-2: the
clone collected them unchanged, before and during the critic-only iteration.
Its end is the mean over iterations 91-100. The cell **learns** if the end
exceeds the start by at least **+0.2** on at least **2 of 3** seeds.

---

## Predictions, and what falsifies each

**P1 - the cell runs end to end**: 100 iterations logged, at least 10
checkpoints, no traceback, clients exit 0, on every seed.

**P2 - `fpo-policy` learns square under FPO with FPO++'s square fine-tuning.**

> Grounds: known item 2. FPO++ rises on square under these settings and never
> falls. Known item 3: with them the critic starts learning within two
> iterations, where E37's did not. Against it: +0.2 from about 0.5 is a
> larger rise than FPO++'s own image-based run makes in 8M steps (about
> +0.3 from 0.28), and E39 runs 4.8M. Falsified if fewer than two seeds gain
> 0.2.

**Reported, not predicted:**
- success, explained variance and clipped fraction by window of ten
  iterations
- fifty-episode evaluations of each seed's final checkpoint and of the clone,
  with E35's `eval_bc.sh` as E37's were read
- wall clock

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, V2, the status rule, P2, then the reported figures.
