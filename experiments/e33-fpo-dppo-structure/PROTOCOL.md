# E33 measurement protocol (pre-registered)

**Written 2026-09-27, before any E33 run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E30 found what stops `fpo-policy` under DPPO: the structure it carries from
FPO's playground. With `dppo-policy`'s structure - three hidden layers of
512, four actions per inference, twenty flow steps - and E29's noise level
of 1.0, it learned Hopper in 100 iterations on all three seeds. That
configuration (E30's `all`) was chosen on Hopper, after seeing Hopper.

**Fixed as it is, does it learn all three MuJoCo tasks - including Hopper
on seeds the choice was not made with?**

### What this cannot settle

* Whether a better configuration exists. `all` is taken unchanged; nothing
  is tuned per task.
* Why the structure matters (E30's open question).

---

## Declared in advance: what was already known

1. **E30's `all`, Hopper, seeds 0 / 1 / 2**: iterations 91-100 at 762 / 873
   / 989, from 20 / 20 / 21, still climbing steeply; logged entropy -0.0040
   at every iteration; `approx_kl` about 5 x 10⁻⁷ over its first ten
   iterations, 3.5 x 10⁻⁶ over its last ten.
2. **`fpo-policy` under DPPO with its default structure**: Hopper 25 to 27,
   Walker2d 1 to 3 at 100 iterations (E28); HalfCheetah no learning in
   409,600 steps (E17, E18).
3. **`dppo-policy` under DPPO**, the policy whose structure `all` borrows:
   HalfCheetah +481 / +84 / +544 at 100 iterations (E28, learns on 2 of 3);
   Hopper 873 / 961 / 812 and Walker2d 528 / 622 / 585 at 300 (E31). Walker2d
   passed 500 between iterations 141 and 200.
4. **Runs are not reproducible bit for bit** after the first update (E20;
   E31's Walker2d against E28's).

---

## Design

| cell | task | DPPO variant | seeds | iterations |
| --- | --- | --- | --- | --- |
| `fpo-cheetah` | HalfCheetah-v5 | `dppo cheetah` | 0, 1, 2 | 300 |
| `fpo-hopper` | Hopper-v5 | `dppo hopper` | **3, 4, 5** | 300 |
| `fpo-walker` | Walker2d-v5 | `dppo walker` | 0, 1, 2 | 300 |

Every cell is E30's `all` arm unchanged: `fpo-policy` with
`--policy.action-horizon 4 --policy.flow-steps 20 --policy.hidden-dims 512
512 512`, DPPO with `--algo.sampling-noise-level 1.0
--algo.logprob-noise-level 1.0`, buffer 1,024 chunks and minibatch 512 (4,096
environment steps per iteration), the client replanning every 4 steps;
checkpoints every 20 iterations. E30's `run_cell.sh`, unchanged. The three
DPPO variants are identical configurations (the MuJoCo variants of E24 and
E28). 300 iterations (1,228,800 steps per seed) is E31's length for the
DPPO row, fixed before any data.

Machine `guangzhao`, CPU, one thread per process, the three cells at once;
code this branch (E30 plus this directory), client plugrl-env-client at
931ab56.

---

## Checks

* **V1 - the noise schedule took**: every cell's logged `losses/entropy`, at
  every iteration, within 0.005 of **-0.004**, E30's value for `all`. It
  depends only on the noise schedule and the step count, not on the task or
  the weights. A cell that fails is not read.

---

## The status rule

As the coverage figure and E28 have it, on the mean return of iterations
291-300, on at least **2 of 3** seeds: Hopper and Walker2d at least **500**;
HalfCheetah at least **+200** over its first iteration. A cell that passes
**learns**; one that does not **did not learn in 1,228,800 steps**.

---

## Predictions, and what falsifies each

**P1 - every cell runs end to end** (300 iterations logged, at least 15
checkpoints, no traceback, client exit 0, on every seed).

**P2 - `fpo-policy` learns Hopper under DPPO on seeds 3, 4, 5.**

> Grounds: known item 1 - at 100 iterations every seed was past 750 and
> rising. Falsified if fewer than two seeds reach 500.

**P3 - `fpo-policy` learns Walker2d under DPPO.**

> Grounds: `dppo-policy`, whose structure this is, learns it in 300
> iterations (known item 3), and with it `fpo-policy` learned Hopper faster
> than `dppo-policy` did. Falsified if fewer than two seeds reach 500.

**P4 - `fpo-policy` learns HalfCheetah under DPPO.**

> Grounds: `dppo-policy` learns it by +200 in 100 iterations on 2 of 3
> (known item 3). Falsified if fewer than two seeds gain 200.

**Reported, not predicted:** the mean return over iterations 91-100 (E30's
length), 191-200 and 291-300 for every seed; wall clock.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.

---

## Reading order

P1, V1, the status rule, P2, P3, P4, then the reported figures.
