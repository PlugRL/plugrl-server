# E57 measurement protocol (pre-registered)

**Written 2026-10-03, after the pilot in `results/pilot/`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E54 and E56 judged each run by what the trainer saw: its own log, its
weights, or the two sides' records. None of them asks what a faulted run
leaves behind, the policy. Reviewers asked whether the faults matter at all.

**Evaluated on a clean environment, is a policy trained through a fault
worse than one trained without it? In particular, are the faults that both
curve checks miss?** And does the trainer's log agree with the clean
evaluation?

---

## Declared in advance: what was already known

1. **The policies.** Every E54 and E56 run saved its final policy
   (`model.zip`): 590 runs of E54 (Pendulum, HalfCheetah) and 244 of E56
   (Hopper). Their outcomes under the gain threshold, seed spread, byte
   identity, record and ledger are published (`runs.csv`, `ledger.csv`). No
   policy had been evaluated outside training before the pilot.
2. **The pilot** (`results/pilot/`), 20 episodes each, on guangzhao:

   | env | `none` seed 0 | `none` seed 10 | `obs:f16` 1.0 seed 0 |
   | --- | --- | --- | --- |
   | Pendulum | -147.0 | -139.1 | -139.7 |
   | HalfCheetah | 647.1 | 779.8 | 851.2 |
   | Hopper | 3368.9 | 3651.3 | 1633.5 (mean length 482) |

   Pendulum's episodes vary by about 100 (initial states); HalfCheetah's by
   about 130; Hopper's by 10-580. An evaluation took 0.3-2.5 s.

---

## Design

`evaluate.py RUN_DIR OUT_DIR --episodes 50`. It does the following:
- loads the run's `model.zip` on CPU, with one torch thread;
- makes a fresh `gymnasium.make(id)` at default settings (Pendulum-v1,
  HalfCheetah-v5, Hopper-v5), with no bridge, client, wrapper or fault;
- plays 50 episodes with `deterministic=True`, reset with seeds 10000-10049,
  which are the same for every policy;
- writes `clean_eval.json`: each episode's return and length, and their mean,
  the **clean return**.

Every run directory of E54 and E56 that holds a `model.zip` is evaluated:
834 runs, 12 at a time on guangzhao (`run.sh`). Then `python summarise.py`
and `python verdicts.py`.

---

## How each run is judged

- **The fault-free band**, per environment: the clean returns of `none` at
  seeds 10-19. The band is its mean and standard deviation.
- **Harmed:** a run's clean return is more than 3 standard deviations below
  the band's mean. The rule is the seed-spread check's, applied to the clean
  return, and only on the low side. A fault-free seed among 10-19 is judged
  against the other nine.
- **Silent on the curve:** the run passed both the gain threshold and seed
  spread (`rule` = 0 and `strict` = 0 in E54's or E56's `runs.csv`).
- **The log's error:** the trainer's logged final return (`last` in
  `runs.csv`) minus the clean return.

Boundary faults are E54's 21 client faults (23 on Hopper). Log faults,
controls (`none`, `delay-1ms`, `wire-float64`) and the two loud faults
(`indices:reorder`, `reward:short-array`) are reported separately. The
loud faults raised an error, so they are left out of every count.

---

## Checks that the runs are valid

- **V1:** all 834 evaluations complete, each with 50 episodes.
- **V2:** two runs with the same weights hash get the same clean return. The
  controls and log faults ended on `none`'s weights, so this tests that
  evaluation is deterministic.

---

## Predictions, and what falsifies each

**P1 - The harm rule raises few false alarms.** It flags at most 1 of the
ten fault-free seeds 10-19 in each environment, each judged against the
other nine. Falsified by 2 or more in any environment.

**P2 - One changed value does no measurable harm.** At dose `one`, at most
10% of boundary-fault runs are harmed, in each environment. Falsified by
more.

**P3 - A zeroed reward leaves a policy that does not work, behind a log
that looks perfect.** On Pendulum at dose 1.0, `reward:zero` is harmed at
all three seeds, while its log reads 0, the best return Pendulum can give.
Falsified by any seed not harmed.

> Grounds: P1, E54's and E56's seed-spread false alarms (0, 1, 0 of ten). P2,
> one changed value moved the logged return about as much as a change of
> seed (E54, E56). P3, with every reward zero PPO's advantages are zero
> after normalisation, and nothing is learned.

**Reported, not predicted:**
- **The main count.** Among boundary-fault runs, at each dose:
  - how many are harmed;
  - how many are harmed while silent on the curve.
- **The faults behind it.** Every fault and dose that was silent on the curve
  at all three seeds and harmed at two or more.
- **The log's error.** For each kind, against the fault-free runs' range.
- **The pilot's lead.** Whether Hopper's `obs:f16` at 1.0 is harmed at seeds 1
  and 2. The pilot saw it at seed 0 only.
