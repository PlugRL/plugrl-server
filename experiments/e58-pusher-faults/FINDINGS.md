# E58: on a 7-joint arm (Pusher), E54-E57 hold: the curves miss most faults, the step ledger catches and locates every one, and no fault the curves miss leaves a worse policy

2026-10-03 · guangzhao (E54's `bridges-venv`) · protocol:
[`PROTOCOL.md`](PROTOCOL.md), after the pilots in `results/pilot/`, before
the registered runs · no amendment

---

## The result

E54's 21 client faults and 2 log faults on Pusher-v5 were run at doses one,
0.01 and 1.0, 409,600 steps each:
- seeds 0-2 were traced, ledgered and evaluated on a clean environment;
- seeds 10-19 were run fault-free;
- 226 runs in all.

Client-fault runs, 63 at each dose:

| dose | status rule | strict | byte identity | record | step ledger | harmed (clean) |
| --- | --- | --- | --- | --- | --- | --- |
| one value | 0 | 0 | 63 | 18 | **63** | 0 |
| 0.01 | 3 | 1 | 63 | 18 | **63** | 3 |
| 1.0 | 21 | 31 | 63 | 15 | **63** | 27 |

Every check and prediction holds (`results/verdicts.txt`):
- **V1:** all 226 runs completed. **V2:** the controls ended on `none`'s
  weights at every seed.
- **P1:** byte identity caught 189 of 189 client-fault runs. This includes
  the `final-obs` faults: every Pusher episode ends at the time limit, where
  SB3 reads the final observation.
- **P2:** the log faults left the weights alone in 18 of 18 runs.
- **P3:** the status rule caught 0 of 69 runs at one value, 3 of 69 at 0.01
  and 21 of 69 at 1.0 (caps 10%, 10%, 60%).
- **P4:** the record caught every run it was predicted to and none it was
  predicted to miss.
- **P5:** the bridge's log caught 18 of 18 log-fault runs and 0 of 189
  client-fault runs.
- **P6:** the ledger caught and located 189 of 189 client-fault runs. It
  found no difference in 27 of 27 control and log-fault runs.
- **P7:** the ledger or the record caught 207 of 207 runs.
- **P8:** the clean harm rule flagged 0 of the 10 fault-free seeds.
- **P9:** at doses one and 0.01, 3 of 126 runs were harmed.

---

## What it means

**On an arm, as on the three earlier tasks, the curves miss most faults.**
- One changed value moved the logged final return by a median of 0.7. The
  fault-free seeds' standard deviation is 1.9.
- At one value, neither curve check caught any of the 63 runs.
- At every value, 10 faults still passed both curve checks at all three
  seeds:
  - `obs:f16` and `obs:stale`;
  - `action:f16` and `action:stale`;
  - `reward:f16` and `reward:stale`;
  - all four `final-obs` faults.

**The step ledger caught and located all of them**, with no fault-free run.

**No fault that the curves missed left a worse policy.** On a clean
environment, 0 of the 155 runs that passed both curve checks were harmed.
The faults harmed 30 runs, and a curve check had already caught each one:
- `terminated:spurious` at 0.01;
- `action:zero`, `action:swap`, `obs:zero`, `obs:swap` and others at every
  value.

The runs that passed both curve checks had a mean clean return within 0.2
band standard deviations of the fault-free band at every dose. This is
E57's result again, on a robot arm.

**The log can lie in plain sight.** With every reward zeroed, the trainer's
log read 0 at every seed. Pusher's rewards are never positive, so 0 is the
best possible return. Yet the policies scored -99, -118 and -129 on a clean
environment, against a band of -31.8 ± 2.1. The status rule caught these
runs, because their log did not rise.

---

## What E58 does not show

* **A real robot, or images.** Pusher is a simulated arm with state
  observations.
* **Any trainer but SB3's PPO.**
* **Across machines.** E58 ran on one machine.

Kept here, in `results/`:
- `runs.csv`, `by_dose.csv` and `summarise.out`, from `summarise.py`;
- `ledger.csv`;
- `runs.jsonl`, from `merge_runs.py`;
- `verdicts.txt`, from `verdicts.py`;
- `clean_evals.json`: each run's clean return, without the 50 episode returns;
- the source hashes.

The per-run directories, with each run's 50 clean episodes, are kept on
guangzhao. A run on one machine is deterministic, so rerunning its line of
`jobs.txt` reproduces it.
