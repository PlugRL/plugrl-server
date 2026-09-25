# E23 measurement protocol (pre-registered)

**Written 2026-09-25, after the pilot in `results/pilot.txt` and before any
registered run.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

Every learning curve this platform has produced on a CPU is HalfCheetah's (E6,
E16 to E20). HalfCheetah never terminates: every episode is cut at 1,000
steps. So one path through the platform has never run inside a learning run -
an episode that **ends because the agent failed**, where the client reports
`terminated`, the server must record it as such, and GAE must not bootstrap
past it.

Hopper-v5 terminates whenever the hopper falls, which an untrained policy does
within about 21 steps. **Does FPO, on this platform, learn a task whose
episodes terminate?**

It is also a heavier test of episode bookkeeping than anything so far. An
untrained Hopper policy ends about 195 episodes per 4,096-step iteration,
against about 4 on HalfCheetah.

### What reading the code already says

`GAEBuffer.compute_advantages_and_returns`, with FPO's default
`treat_truncated_as_done=False`, handles the two endings differently:

* **terminated:** `next_non_terminal = 0`, so no bootstrap past the last
  step. Correct.
* **truncated:** `trunc_mask = 0`, so that step's `δ` and advantage are zeroed
  and the GAE chain is cut there. The final transition is dropped rather than
  bootstrapped. That is a defensible choice, and HalfCheetah has run on it all
  along.

That is reasoning, not a measurement. E23 is the measurement, end to end.

### What this cannot settle

* Anything about FPO against other algorithms on Hopper. There is no
  comparison arm, and FPO's defaults were never tuned for this task.
* If FPO does not learn, E23 does not say why. By the project's rule, a
  platform that fails to learn is a defect in the platform until shown
  otherwise. The next step would be controlled tests of the termination path,
  not a hyperparameter search.

---

## Declared in advance: what was already known

1. **The pilot** (`results/pilot.txt`): one iteration per seed, collected
   before any update. First-iteration return **16.25, 19.05, 14.48**, mean
   episode length **21.0, 22.4, 20.4**. The harness started and ended cleanly
   on all three seeds.
2. **The same line on HalfCheetah**, E16's Phase A on this code: a first
   update of **−328.1** averaged over seeds (`compare_e6.py`, single updates),
   and **1453.6, 2049.5, 1345.7** as the mean of iterations 91 to 100 - the
   measure `end` uses here (E16's P1 table). 91 minutes for three concurrent
   seeds.
3. **Run-to-run noise.** Learning here is not deterministic past the first
   update (E20). Every threshold below is far from the null, so none depends
   on reproducibility.
4. **The Hopper-v5 reward.** The environment gives 1 per step alive, plus
   forward velocity, minus a small control cost. A policy that only learns to
   stand still until the 1,000-step cut scores about 1,000. A policy that has
   learned nothing scores about 17.

---

## Held fixed

E16 Phase A exactly, except the environment and the two dimensions it
forces: `fpo-policy` defaults on CPU with `--policy.obs-dim 11
--policy.action-dim 3`; `fpo` defaults; `mujoco-v1` with `--env.name
Hopper-v5`, one env per client; `--runner.replan-steps 1`; server and client
seeds 0, 1, 2; buffer 4,096; 409,600 steps, which is 100 iterations; three
runs at a time; checkpoints every 20 iterations.

The client's episode budget is the step budget, not E16's `STEPS / 1000 + 10`.
That count assumes 1,000-step episodes and would end a Hopper client within
its first few iterations (`run.sh`).

---

## What is measured

`rollout/reward` and `rollout/length` per iteration, from each run's
tensorboard. **`start`** is the first iteration. **`end`** is the mean of
iterations 91 to 100.

---

## Predictions, and what falsifies each

**P1 - the platform completes the runs.** On every seed: 100 iterations
logged, a checkpoint within one update of 409,600, no traceback in the
server log, and the client exits 0.

> This is the bookkeeping test: about 195 episode ends per iteration at the
> start, each one a terminated flag, a reset, and a GAE boundary. Falsified
> by any seed failing any part. If P1 fails, P2 is not read.

**P2 - FPO learns Hopper.** On at least **2 of 3** seeds, `end` is at least
**500**.

> Checked against the null before being written down: the untrained policy
> scores 14 to 19, and a policy that has learned nothing cannot reach 500 at
> any noise level seen on this line. 500 is well short of a policy that has
> learned to stand (about 1,000), so it asks for substantial learning and not
> for mastery. Falsified by two or more seeds below 500.

**P3 - wall clock.** Under **150 minutes** for the three concurrent seeds,
E16's bound for the same line.

> Hopper's physics is cheaper per step than HalfCheetah's, and its short early
> episodes add resets. Falsified by exceeding it.

**Reported, not predicted:** `end − start` and the final mean episode length
per seed, and whether any seed reaches `end` of 1,000 or more - past standing
still, in which case it is also moving forward.

---

## Declared deviations allowed in advance

1. One restart of any run that dies for a reason outside the experiment,
   recorded in `AMENDMENT.md`.
2. The machine is shared with other work. If another CPU-heavy job starts
   during the runs, it is recorded in `AMENDMENT.md` and P3 is not read.
   P1 and P2 still are.

---

## Reading order

P1, then P3, then P2, then the descriptive figures.
