# E54: a fault at the boundary passes the status rule almost always; byte-identical weights or reconciliation with an env-side record catches every one

2026-10-02 · guangzhao (bridges-venv: torch 2.14.1+cpu, SB3 2.9.0, gymnasium
1.3.0, mujoco 3.14.0), 10 runs at a time · protocol:
[`PROTOCOL.md`](PROTOCOL.md), after the pilots in `results/pilot/`, before
the registered runs · no amendment

---

## The result

590 registered runs. SB3's PPO trained Pendulum and HalfCheetah through the
bridge, as in E50, while the env client changed one field of what crossed,
in one way. There were 21 client faults and 2 faults in the trainer's log,
at doses from one value to every value.

**Share of runs with a changed value caught by each check**, seeds 0-2,
client and log faults together:

| env | dose | runs | status rule | strict | weights | record | weights or record |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pendulum | one value | 69 | 0% | 0% | 91% | 28% | **100%** |
| | 0.1% | 63 | 0% | 8% | 100% | 29% | **100%** |
| | 1% | 69 | 4% | 6% | 91% | 35% | **100%** |
| | 10% | 69 | 4% | 19% | 91% | 35% | **100%** |
| | every value | 69 | 39% | 55% | 91% | 30% | **100%** |
| HalfCheetah | one value | 69 | 4% | 4% | 91% | 33% | **100%** |
| | 1% | 69 | 9% | 9% | 91% | 35% | **100%** |
| | every value | 69 | 32% | 46% | 91% | 33% | **100%** |

The weights miss exactly the log faults, at 9% of the runs. Counting client
faults alone, they caught 100% at every dose on both envs.

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** all 590 runs completed but the loud faults', which raised.
- **V2:** the fault-free runs ended on E50's weights at seeds 0-2, for both
  envs (`ec5b8226`, `062e92e3`, `3a8d968f`; `8261c03a`, `86564694`,
  `65cc8843`), with 10 runs at a time on the machine.
- **P1:** the weights differed in all 504 client-fault runs with a changed
  value.
- **P2:** they were the fault-free weights in all 48 control and log-fault
  runs.
- **P3:** on Pendulum the status rule caught 0, 0, 3, 3 and 27 of 69, 63,
  69, 69 and 69 runs at the five doses, under the caps of 10% and 60%.
- **P4:** the record caught every run it was predicted to catch, and none it
  was predicted not to.
- **P5:** the bridge's own log caught all 42 log-fault runs and none of the
  504 client-fault runs.
- **P6:** weights or record caught all 546 silent runs with a changed value.
- **P7:** both loud faults raised at every seed, about 9.5 s into the run.
- **P8:** the first difference located all 351 traced Pendulum runs: their
  step, env and field.
- **P9:** on HalfCheetah the status rule caught 3 and 6 of 69 at one value
  and 1%.

---

## What it means

**A curve cannot tell a fault from a seed.** One changed value in 102,400
moved Pendulum's final return from its seed's fault-free run by a median of
14, and HalfCheetah's by a median of 67. The fault-free runs at seeds 10-19
spread with a standard deviation of 11 and 146. So one value changes the
outcome about as much as a change of seed. A run with a fault was at least
as likely to look better as worse: at one value, 43 of 63 Pendulum runs and
33 of 60 HalfCheetah runs ended above the fault-free run at their own seed.
- The status rule asks only whether the policy learned at all. It caught a
  fault in two cases:
  - when the fault broke learning: zeroed or swapped observations, actions
    or rewards at every value, and spurious terminations;
  - when it left the log with no episodes: every step skipped, every time
    limit dropped.
- The strict check, against ten fault-free seeds, had few false alarms (0
  of 10 and 1 of 10 seeds judged against the other nine). It caught little
  more, because it can only see what moves a final return beyond the seeds'
  spread.
- Some faults made the log look better. With every reward zeroed, Pendulum
  logged its best possible return, 0. Spurious terminations cut Pendulum's
  episodes short, and short episodes lose less.

**Byte identity sees every change to what training receives, and nothing
else.** A fault changed the weights whenever it changed one value that
crossed, at every dose and on both envs. Time (`delay-1ms`) and
representation (`wire-float64`) did not change them, and neither did a
fault in the log.

**Reconciliation sees what changes the books.** The trainer's log and the
client's record disagreed whenever a fault changed:
- a reward, beyond the record's tolerance;
- an episode end;
- a step;
- the log itself.

Faults in observations, actions and final observations, and a time limit
sent as a termination, left the episodes as the environment produced them.
It could not see those. It could not see a reward changed by less than its
tolerance either: `reward:f16` at any dose, or `reward:zero` and
`reward:stale` at one value.

**The two checks cover each other.** Between them they caught every one of
the 546 silent runs. The two faults the bridge rejects raised at once.

**Where the record is kept matters.** The bridge's own log counts the
rewards as they arrive on the trainer's side of the wire. It caught every
fault in the trainer's log and no fault in the client's: by the time a
client fault reaches the bridge, the bridge counts what it was sent. A
record that is to catch a client's faults has to be kept by the client,
next to the environment.

**A fault can be located.** On one machine the fault-free run is
deterministic, so a faulted run crosses the same bytes up to the fault. The
first difference named the fault's step, env and field in all 351 traced
runs.

---

## What E54 does not show

* **Faults on any trainer but SB3, or any env but these two.** Both envs
  are forgiving: a policy still learns through most faults.
* **Byte identity across machines.** It needs a fault-free run on the same
  machine (E50). E55 tests a check that does not.
* **Faults the protocol's checkers would catch.** Only the bridge's own
  validation was in the loop.
* **Faults chosen at random.** The cells were; the faults were chosen to
  cover every field that crosses and the ways it goes wrong in practice. The
  paper maps each to a fault found in a real library.

Kept here, in `results/`:
- `runs.jsonl`: every run's `result.json` and `fault.json`, and the last
  line of its logs (`merge_runs.py`);
- `runs.csv`: every check's verdict on every run (`summarise.py`), and
  `by_dose.csv`, `summary.txt`;
- `localisation.csv` (`localise.py`) and `verdicts.txt` (`verdicts.py`);
- `jobs.txt`, `registered.out`, `load.txt`, and the sources' hashes.

The per-run directories are kept on guangzhao and not committed:
- the per-episode logs behind the record checks (`monitor.csv`,
  `episodes.csv`, `env_record.csv`: 100 MB, 18 MB compressed);
- the traces and the final policies.

A run is deterministic on its machine, so rerunning its line of `jobs.txt`
there reproduces them byte for byte.
