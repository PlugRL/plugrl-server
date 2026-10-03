# E56: on Hopper, which ends episodes when it falls, the step ledger still catches and locates every fault in what crosses, and a learning curve still misses most of them

2026-10-02 · guangzhao (E54's `bridges-venv`) · protocol:
[`PROTOCOL.md`](PROTOCOL.md), after the pilots in `results/pilot/`, before
the registered runs · no amendment

`verdicts.py`, which judges the predictions as `PROTOCOL.md` words them, was
not in the protocol's commit. It was last changed at 21:12 EDT. That is
after the registered runs started (21:10) and before the first one ended
(21:15). It is committed here unchanged.

---

## The result

E54's 21 client faults plus two in terminations (`terminated:drop`,
`terminated:as-trunc`) and E54's two log faults, on Hopper-v5 at doses one,
0.01 and 1.0. Seeds 0-2 traced and ledgered, and seeds 10-19 fault-free for
the strict check's band. Each run is 1,024,000 steps.

Client-fault runs with a changed value, 69 at each dose:

| dose | status rule | strict | byte identity | record | step ledger |
| --- | --- | --- | --- | --- | --- |
| one value | 0 | 0 | 57 | 19 | **69** |
| 0.01 | 3 | 3 | 69 | 21 | **69** |
| 1.0 | 33 | 34 | 69 | 18 | **69** |

The ledger found no difference in any of the 27 control and log-fault runs.
The strict check flagged none of the ten fault-free seeds, each judged
against the other nine.

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** all 244 runs reached 1,024,000 steps; every client exited 0.
- **V2:** `delay-1ms` and `wire-float64` ended on `none`'s weights at seeds
  0-2.
- **P1:** byte identity caught 171 of 171 client-fault runs with a changed
  value, outside the `final-obs` faults.
- **P2:** the log faults ended on `none`'s weights in 18 of 18 runs.
- **P3:** the status rule caught 0 of 75 runs at one value, 3 of 75 at 0.01
  and 33 of 75 at 1.0 (caps 10%, 10%, 60%).
- **P4:** the record caught every run it was predicted to catch and none it
  was predicted to miss.
- **P5:** the bridge's log caught 18 of 18 log-fault runs and 0 of 207
  client-fault runs.
- **P6:** the ledger caught 207 of 207 client-fault runs, and each first
  difference was the fault's first changed cell; no difference in 27 of 27
  control and log-fault runs.
- **P7:** the ledger or the record caught 225 of 225 runs with a changed
  value.

---

## What it means

**E54's and E55's findings hold on an environment that ends episodes
early.**
- One changed value moved the final return by a median of 197. The fault-free
  seeds' standard deviation is 259. Neither curve check caught any of the 69.
- At dose 1.0, every value changed, and 11 faults still passed both curve
  checks at all three seeds. Among them were `obs:f16`, `action:stale`,
  `reward:stale`, `truncated:drop` and all four `final-obs` faults.
- The ledger caught and located all 207 client-fault runs, with no fault-free
  run to compare against.

**A fault can cross without changing the weights, and only the ledger sees
it.** At one value, each `final-obs` fault changed the final observation of
an episode that ended in a fall, at vector step 3,255-3,379, env 0. SB3
bootstraps from a final observation only at a time limit, so PPO never read
the changed value. The weights were `none`'s in 12 of 12 runs. This was
expected from the pilot and left out of P1. The ledger caught and located
all 12.

The value is read again as soon as the same fault lands at a time limit. At
doses 0.01 and 1.0 it does, and byte identity caught 24 of 24. A check that
looks only at the weights says "no fault" about a boundary that is
corrupting final observations. That is right for this run and this trainer,
and wrong for any trainer that reads them.

**Faults in terminations are silent too, and they behave unlike each other.**
- `terminated:drop` loses a fall. At 0.01 it passed both curve checks at
  every seed, and the record and the ledger caught it.
- `terminated:as-trunc` sends a fall as a time limit. It changes nothing in
  the record's books, so the record missed it at every dose. The ledger
  caught it every time, and so did byte identity.
- At 1.0 the status rule caught all six runs. `terminated:as-trunc` broke
  learning. With `terminated:drop`, the trainer's log held no episode at
  all, while the environment's record held 7,178 at seed 0.
- `truncated:as-term` and `truncated:drop` passed both curve checks at every
  dose.

---

## What E56 does not show

* **Any trainer but SB3's PPO.** Whether a fault in final observations
  reaches the weights depends on how a trainer bootstraps. E56 shows one
  trainer.
* **Faults inside the environment.** The ledger compares what crosses. A
  fault before the wrapper is beyond it, as in E55.
* **A ledger checked while the run goes on.** The two sides were compared
  after each run.
* **Across machines.** E55 showed the ledger across two machines on
  Pendulum; E56 ran on one.

Kept here, in `results/`: `ledger.csv` (`ledger.py`), `runs.csv`,
`by_dose.csv` and `summarise.out` (`summarise.py`), `runs.jsonl`
(`merge_runs.py`, every run's `result.json` and `fault.json`), and
`verdicts.txt` (`verdicts.py`), with the source hashes. The per-run directories are kept
on guangzhao and not committed. Each one holds the episode logs, the env
record, the final policy and `ledger.json`. Each traced run's trace and
ledger arrays were deleted once judged, as `run.sh` says: they are 150 MB a
run. A run on one machine is deterministic, so rerunning its line of
`jobs.txt` reproduces them.
