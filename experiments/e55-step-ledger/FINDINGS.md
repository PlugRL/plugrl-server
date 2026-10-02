# E55: a step ledger on both sides of the boundary catches and locates every fault in what crosses, with no reference run, across machines too

2026-10-02 · guangzhao (E54's `bridges-venv`) and a Windows 11 laptop
(Python 3.11.14, numpy 2.3.2, gymnasium 1.3.0) over Tailscale · protocol:
[`PROTOCOL.md`](PROTOCOL.md), after the pilots in `results/pilot/`, before
the registered runs · no amendment

---

## The result

E54's faults on Pendulum, at doses one, 0.01 and 1.0. Each environment was
wrapped in `LedgerEnv`, which records what it took and gave. The trainer
recorded what it sent and received. `ledger.py` compared the two, step by
step, with no fault-free run.

| | runs | client faults caught and located | false alarms | ledger or record |
| --- | --- | --- | --- | --- |
| one machine, seeds 0-2 | 216 | **189 of 189** | 0 of 27 | 207 of 207 |
| across machines, seed 0 | 24 | **21 of 21** | 0 of 3 | 23 of 23 |

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** all 240 runs completed; every client exited 0.
- **V2:** `none` ended on E50's weights at seeds 0-2. The wrapper changes
  nothing that crosses.
- **P1:** on one machine, all 189 client-fault runs with a changed value
  were caught. Each first difference was the fault's first changed cell,
  in the field the fault touches.
- **P2:** no difference in any of the 30 control and log-fault runs, on one
  machine or across two.
- **P3:** across machines, all 21 client faults at one value were caught
  and located; `none` showed no difference.
- **P4:** across machines, `none` ended on `d7a19cc0`, not guangzhao's
  `ec5b8226`.
- **P5:** the ledger or E54's record caught every run with a changed value:
  207 of 207, and 23 of 23.

---

## What it means

**Compare the two sides, and nothing else is needed.** A boundary fault is a
difference between what the environment took and gave and what the
trainer sent and received. Recording both and comparing them finds the
fault where it is: the step, the env and the field. The ledger counted
exactly the values a fault changed: 1,068 for `obs:f16` at 1%, the same
count the client kept. It does not need a fault-free run, a seed that
reproduces, or the two sides on one machine.

**Across machines it is the only check of the four that sees every fault
in what crosses.**
- Byte identity cannot be used there. A fault-free run across machines
  ended on other weights than the same seed on guangzhao, as E43 and E50
  found: it would have called that run faulty.
- The status rule caught none of the 24 runs.
- E54's record caught 8, the faults that change the books.
- The ledger caught all 21 client faults and raised no alarm on the three
  others.

**What it leaves to the record.** A fault in the trainer's own log changes
nothing that crosses, so the ledger cannot see it. E54's record, kept at the
environment, catches those. The two together caught every fault, on one
machine and across two.

**What it costs.** The ledger is a wrapper around each environment, plus a
trace on the trainer's side. Neither changed a byte of what crossed (V2).
A run across machines took 42-55 s, against 20 s on one machine. That is
the link, not the ledger: E43 measured the crossing's cost.

---

## What E55 does not show

* **A ledger checked while the run goes on.** The two sides were compared
  after each run. A digest of each step carried in the protocol would let
  the server check as it goes; that is a protocol feature still to design.
* **Faults inside the environment, or before the wrapper.** The ledger sits
  at the environment's own interface. A fault there, or in an environment
  reached only through its own client, is beyond it.
* **Any env but Pendulum, or any trainer but SB3.** E54 covers HalfCheetah
  for the other checks.
* **Faults it was not built to see.** Every field it compares is one E54
  injected a fault into; a field that crosses and is not compared
  (`info`) is not checked.

Kept here, in `results/` and `results/cross/`: `ledger.csv` (`ledger.py`),
`runs.csv` and `summary.txt` (`summarise.py`), `runs.jsonl` (`merge_runs.py`,
every run's `result.json` and `fault.json`), and `verdicts.txt`
(`verdicts.py`). The per-run directories are kept on guangzhao and the
laptop and not committed: traces and ledgers, the episode logs and the
final policies. A run on one machine is deterministic, so rerunning its line
of `jobs.txt` there reproduces them.
