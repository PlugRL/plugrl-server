# E57: on a clean environment, the policies behind the faults that both curve checks miss are as good as fault-free ones; every policy the faults wrecked, the curve checks had already caught

2026-10-03 · guangzhao (E54's `bridges-venv`) · protocol:
[`PROTOCOL.md`](PROTOCOL.md), after the pilot in `results/pilot/`, before
the registered runs · amended once, see [`AMENDMENT.md`](AMENDMENT.md): a
parsing fix in `summarise.py`, made before any output, that changes no count

---

## The result

Each of the 834 final policies of E54 and E56 was run on a fresh gymnasium
environment for 50 deterministic episodes, all policies on the same seeds.
"Harmed" means its mean return, the clean return, fell more than 3 standard
deviations below the fault-free band (`none`, seeds 10-19).

Boundary-fault runs, seeds 0-2:

| env | dose | harmed | silent on the curve | of them harmed |
| --- | --- | --- | --- | --- |
| Pendulum | one | 1/63 | 63 | 1 |
| | 0.001 | 2/63 | 58 | 1 |
| | 0.01 | 1/63 | 59 | 1 |
| | 0.1 | 6/63 | 50 | 2 |
| | 1.0 | 24/63 | 25 | 0 |
| HalfCheetah | one | 0/63 | 60 | 0 |
| | 0.01 | 0/63 | 57 | 0 |
| | 1.0 | 28/63 | 31 | 0 |
| Hopper | one | 0/69 | 69 | 0 |
| | 0.01 | 0/69 | 66 | 0 |
| | 1.0 | 19/69 | 35 | 0 |

"Silent on the curve" means the run passed both the gain threshold and
seed spread.

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** 834 of 834 evaluations, each of 50 episodes.
- **V2:** the 12 weights shared by two or more runs got identical clean
  returns, so evaluation is deterministic.
- **P1:** the harm rule flagged none of the ten fault-free band seeds in any
  environment.
- **P2:** at one value, 1 of 63, 0 of 63 and 0 of 69 boundary-fault runs
  were harmed.
- **P3:** Pendulum's `reward:zero` at every value logged 0, the best return
  Pendulum gives, at all three seeds. Its policies scored -1196, -1550 and
  -1639 on the clean environment, against a band of -186 ± 26.

---

## What it means

**Every policy the faults wrecked, the curve checks had already caught.** At
every value, the faults harmed 24, 28 and 19 runs. None of them passed
both curve checks. At HalfCheetah and Hopper, no run that passed both checks
was harmed, at any dose.

**The faults the curve checks miss leave policies as good as fault-free
ones.** This is an exploratory analysis, not in the protocol (`explore.py`,
`results/explore.txt`). For each environment and dose, the runs silent on
the curve have a mean clean return 0.03-0.65 band s.d. *above* the band.
None is significantly below the 13 fault-free runs (two-sided Mann-Whitney
U). The one significant difference is upward: Pendulum at every value,
p = 0.003. At their own seed, they beat the fault-free twin by a median of 7-9 on
Pendulum, 63-101 on HalfCheetah and 142-368 on Hopper. That is consistent with a
change of seed, not with harm.

**The exception is five Pendulum runs.** They passed both curve checks but
scored below every one of the 13 fault-free runs, whose worst was -238:
- `reward:f16` at 0.1%: -744;
- `obs:swap` at 1%: -506;
- `obs:stale` at one value: -430;
- `final-obs:f16` at 10%: -333;
- `obs:stale` at 10%: -277.

That is 5 of 255 silent runs on Pendulum, against 0 of 13 fault-free.
Thirteen fault-free runs cannot tell a rare harm from a rare bad seed.

**So a silent fault is silent on a clean evaluation too.** In these tasks,
with SB3's PPO, a fault that leaves the curve looking right also leaves
the policy looking right. What it changes is the run. The weights differ
from the run the authors think they trained, and with log faults or reward
faults, so do the numbers they report. Pendulum's `reward:zero` is the
extreme case: its log read 0, the best possible, while its policy scored
-1196 to -1639. Only a check of the boundary itself, the step ledger or the
episode record, tells such a run from a correct one.

**On trusting the trainer's log.** For each run, the logged final return
minus the clean return is in `results/summary.txt`. The fault-free runs'
own gap is wide: Hopper's ranges from -1667 to +806. Training returns come
from stochastic actions and the clean return from deterministic ones. A
gap this wide hides all but gross reward faults.

---

## What E57 does not show

* **Any trainer but SB3's PPO, or any task but these three.** On a task
  where small changes in the data matter more, such as sparse rewards,
  long horizons or real robots, the same faults may harm the policy.
* **Harm too rare for 13 fault-free runs to bound.** The five Pendulum
  outliers above are the edge of what this design can see.
* **Harm in a comparison between methods.** E57 compares policies with the
  fault-free band, not one method with another.

Kept here, in `results/`: `clean.csv` (`summarise.py`, one row per run, with
its clean return, harm, curve checks and logged return), `summary.txt`,
`verdicts.txt` (`verdicts.py`), `explore.txt` (`explore.py`), `jobs.txt`,
`load.txt`, `registered.out` and the source hashes. Each run's
`clean_eval.json`, with all 50 episodes, is kept on guangzhao. Evaluation
is deterministic (V2), so `run.sh` reproduces it.
