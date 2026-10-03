# E58 measurement protocol (pre-registered)

**Written 2026-10-03, after the pilots in `results/pilot/`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E54-E57 used Pendulum, HalfCheetah and Hopper. Reviewers asked for a task
closer to robot learning. Pusher-v5 is a 7-joint arm that pushes an object
to a goal, and its episodes all end at a 100-step time limit.

**On a robot arm, do E54-E57's findings hold?**
- **The curves:** do the gain threshold and seed spread still miss most
  faults?
- **The step ledger:** does it still catch and locate every fault in what
  crosses?
- **The clean evaluation:** are the policies behind the faults the curves
  miss as good as fault-free ones?

---

## Declared in advance: what was already known

1. **The code is E56's, with these changes:**
   - `train.py` adds `pusher`: Pusher-v5 at SB3's defaults, with 16 x 256
     frames per rollout, for 409,600 steps, as HalfCheetah.
   - `jobs.py` runs Pusher. It leaves out E56's `terminated:drop` and
     `terminated:as-trunc`: Pusher never terminates, so they have nothing to
     change.
   - `summarise.py` gives Pusher Hopper's status rule: the mean return of the
     last tenth of the trainer's log must beat the first tenth's by at least
     **36**.
   - `run.sh` evaluates every run's final policy on a clean environment
     (`evaluate.py`, E57's, with Pusher-v5): 50 deterministic episodes, seeds
     10000-10049.
   - `verdicts.py` is E56's, with three changes:
     - P1 covers every client fault;
     - P4 drops the two termination faults;
     - P8 and P9 judge the clean evaluation.
2. **The budget pilot** (`results/pilot/budget.txt`): fault-free seeds 8 and
   9, 2,048,000 steps, two at a time.
   - Over the first 409,600 steps, the last tenth of the log beat the first
     tenth by 72.6 and 73.7. The mean return went from about -111 to about
     -38. PPO's schedules are constant, so these first 409,600 steps are the
     same as a 409,600-step run.
   - By 2,048,000 steps the gain was 44.5 and 54.2, and the clean return
     about -27.
   - The bar of 36 is half the gain at 409,600 steps, as Hopper's bar was.
   - A 2,048,000-step run took about 510 s.
3. **The fault pilot**: seed 9, every fault at one value plus the controls,
   409,600 steps, 10 at a time. The outputs are `results/pilot/fault-runs.csv`,
   `fault-ledger.csv` and `fault-by-dose.csv`.
   - `none`, `delay-1ms` and `wire-float64` ended on `124d4c2b`. `none`
     gained 73.7, the same as the budget pilot's first 409,600 steps at seed
     9.
   - The status rule caught none of the 23 faults.
   - The weights caught all 21 client faults, the four `final-obs` faults
     among them. The weights did not catch the two log faults.
   - The record caught 8:
     - `reward:zero`, `reward:stale` and `reward:swap`;
     - `terminated:spurious`, `truncated:drop` and `step:fill`;
     - the two log faults.
   - The bridge log caught the two log faults only.
   - The ledger caught and located all 21 client faults, and found no
     difference in the 5 others: 26 of 26 right.
   - `none`'s clean return was -32.1. A run took about 114 s, 10 at a time.

---

## Design

On guangzhao, `bash run.sh results registered 10`, ports 8920-8929:
- Pusher, seeds 0-2: the 3 controls, and the 21 client faults and 2 log
  faults at doses one, 0.01 and 1.0. All runs are traced, ledgered and
  evaluated. That is 3 x (3 + 69) = 216 runs.
- Pusher, seeds 10-19: fault-free and untraced. They give the strict check's
  band and the clean evaluation's band. That is 10 runs.

226 runs in all. Then `python summarise.py results`, `python merge_runs.py
results`, `python ledger.py results` and `python verdicts.py results`.

---

## How each check is judged

As in E56, with Pusher's status rule above. The clean evaluation is judged
as in E57: a run is **harmed** if its clean return is more than 3 standard
deviations below the mean of the fault-free seeds 10-19.

---

## Checks that the runs are valid

- **V1:** every run reaches 409,600 steps, and its client exits 0.
- **V2:** the two controls end on the same weights as `none` at each of
  seeds 0-2.

---

## Predictions, and what falsifies each

**P1 - Byte identity sees every change.** Every client-fault run with a
changed value ends on weights other than `none`'s at its seed. This
includes the `final-obs` faults: every Pusher episode ends at the time limit,
where SB3 bootstraps from the final observation. Falsified by one run on
`none`'s weights.

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
  - `terminated:spurious`, `truncated:drop`.
- It catches `step:fill` at 0.01, and both log faults at every dose with a
  changed value.
- It catches no run of an `obs`, `action` or `final-obs` fault, or of
  `truncated:as-term`.
- Falsified by one run against any of these.

**P5 - The bridge's log sees only the trainer's side.** It catches every
log-fault run with a changed value and no client-fault run.

**P6 - The ledger catches and locates every fault in what crosses.** Every
client-fault run with a changed value is caught, and its first difference
is right. There is no difference in any control or log-fault run.

**P7 - The ledger and the record together catch every silent fault.**

**P8 - The clean evaluation's harm rule raises few false alarms.** At most
1 of the ten fault-free seeds is flagged, each judged against the other
nine.

**P9 - Low doses do no measurable harm.** At doses one and 0.01, at most 10%
of client-fault runs are harmed.

> Grounds: E54-E57 and the pilots. Every Pusher episode ends at the time
> limit, so its final observations are read; that is why P1 covers
> `final-obs`, unlike E56's. P8 and P9 are E57's P1 and P2.

**Reported, not predicted:**
- the strict check at every dose, and its false alarms;
- how many runs are harmed at each dose, and how many of those were silent
  on both curve checks;
- every run's final return, clean return and wall clock.
