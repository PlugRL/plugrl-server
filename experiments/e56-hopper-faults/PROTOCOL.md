# E56 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilots in `results/pilot/`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E54 and E55 injected faults at the boundary of two forgiving environments,
Pendulum and HalfCheetah, which never end an episode early. So faults in
episode ends hardly mattered there. Hopper ends an episode when it falls,
and it is harder to learn. So:

**On Hopper, do E54's and E55's findings hold?**
- **The status rule:** does it still miss most faults?
- **The two checks:** do byte identity and reconciliation still catch every
  fault between them?
- **The ledger:** does the step ledger still catch and locate every fault in
  what crosses?

**And what do faults in terminations do?**

---

## Declared in advance: what was already known

1. **The code**, E55's with these changes:
   - `faulty_client.py` adds `terminated:drop` (a termination is not sent)
     and `terminated:as-trunc` (a termination is sent as a time limit).
   - `train.py` adds `hopper`: Hopper-v5 at SB3's defaults, with 16 x 256
     frames per rollout, as E50's HalfCheetah, for 1,024,000 steps.
   - `ledger.py` judges one run at a time (`--run`). `run.sh` judges each
     traced run as it ends and deletes its traces: Hopper's are 150 MB a run.
   - `summarise.py` has Hopper's status rule, below.
2. **The faults:** E54's 21 client faults, E56's two above, and E54's two
   log faults, at doses `one`, `0.01` and `1.0`. E54's controls.
3. **The budget pilot** (`results/pilot/budget.txt`):
   - Fault-free Hopper learned on seeds 9 and 10. The mean return of the last
     tenth of the trainer's log beat the first tenth's by 2,016 and 2,169.
   - Seed 9 traced and ledgered ended on the same weights as untraced
     (`2f370852`); its ledger found no difference.
   - A run took about 250 s, three at a time.
4. **The fault pilot**, seed 9, every fault at one value plus the controls,
   10 at a time (`results/pilot/fault-runs.csv`, `fault-ledger.csv`):
   - `none`, `delay-1ms` and `wire-float64` ended on `2f370852`, the budget
     pilot's weights.
   - The status rule caught none of the 25 faults.
   - The weights caught 19 of the 23 client faults. They missed the four
     `final-obs` faults: each changed one final observation, at an episode
     that ended in a termination. SB3 bootstraps from a final observation
     only at a time limit, so PPO never read the changed value.
   - The record caught 9:
     - the three reward faults but `reward:f16`;
     - `terminated:spurious`, `truncated:drop`, `terminated:drop` and
       `step:fill`;
     - the two log faults.
   - The ledger caught and located all 23 client faults, and found no
     difference in the controls and log faults: 28 of 28 right.

---

## Design

On guangzhao, `bash run.sh results registered 10`, ports 8900-8909:
- Hopper, seeds 0-2: the 3 controls, and the 25 faults at the 3 doses, all
  traced and ledgered. That is 3 x (3 + 75) = 234 runs.
- Hopper, seeds 10-19: fault-free, not traced, for the strict check's band.
  That is 10 runs.

244 runs in all. Then `python ledger.py results`, `python summarise.py
results`, `python merge_runs.py results`.

---

## How each check is judged

As in E54 and E55, with Hopper's status rule.
- **rule:** Hopper's episodes vary in length, so the rule counts episodes.
  The mean return of the last tenth of the trainer's log must beat the first
  tenth's by at least **1,000**, about half of what the fault-free pilots
  gained. A log with fewer than 20 episodes fails.
- **strict:** the mean return of the last tenth, against the mean and
  standard deviation of seeds 10-19. More than 3 s.d. away is caught.
- **weights, record, bridge:** as in E54.
- **ledger:** as in E55, including its definition of a right first
  difference. `terminated:drop` shows in the episode end, and
  `terminated:as-trunc` in the time limit.

---

## Checks that the runs are valid

- **V1:** every run reaches 1,024,000 steps, and its client exits 0.
- **V2:** the three controls end on the same weights as `none` at each of
  seeds 0-2. The machine running 10 at a time, the trace and the ledger
  change nothing.

---

## Predictions, and what falsifies each

**P1 - Byte identity sees every change that training reads.** Every run of a
client fault other than the four `final-obs` faults, with a changed value,
ends on weights other than `none`'s at its seed.
- Falsified by one such run on `none`'s weights.
- The `final-obs` faults are reported, not predicted. Their changed values
  matter only where an episode ends at the time limit.

**P2 - and nothing else.** The two log faults end on `none`'s weights at
every seed and dose. Falsified by one that does not.

**P3 - The status rule misses most faults.** Over client- and log-fault runs
with a changed value:
- at doses one and 0.01, the rule catches at most 10% at each dose;
- at dose 1.0, at most 60%.

Falsified by more at any dose.

**P4 - Reconciliation sees what changes the books, and only that.**
- It catches every run, at doses 0.01 and 1.0 with a changed value, of:
  - `reward:zero`, `reward:stale`, `reward:swap`;
  - `terminated:spurious`, `terminated:drop`, `truncated:drop`.
- It catches `step:fill` at 0.01, and both log faults at every dose with a
  changed value.
- It catches no run of any of these:
  - an `obs`, `action` or `final-obs` fault;
  - `truncated:as-term` or `terminated:as-trunc`.
- Falsified by one run against any of these.
- Reported, not predicted:
  - `reward:f16`;
  - every fault at one value;
  - `step:fill` at 1.0.

**P5 - The bridge's log sees only the trainer's side.** It catches every
log-fault run with a changed value and no client-fault run. Falsified by
one run against either.

**P6 - The ledger catches and locates every fault in what crosses.** Every
client-fault run with a changed value is caught, and its first difference
is right. This includes the `final-obs` faults that leave the weights alone.
It finds no difference in any control or log-fault run. Falsified by one run
against either.

**P7 - Between them, the ledger and the record catch every silent fault.**
Every client- and log-fault run with a changed value is caught by one of
them. Falsified by one that neither catches.

> Grounds: the fault pilot, and E54-E55. Hopper's terminations are most of
> its episode ends: that is why the `final-obs` faults are left out of P1,
> and why E56 adds `terminated:drop` and `terminated:as-trunc`.

**Reported, not predicted:**
- the strict check at every dose, and its false alarms on seeds 10-19;
- the `final-obs` faults' weights, at each dose;
- every run's final return and wall clock.
