# E55 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilots in `results/pilot/`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

E54 found two checks that between them catch every fault it injected:
- **byte identity** needs a fault-free run at the same seed on the same
  machine, so it cannot judge a run across machines;
- **reconciliation** with an env-side record of episodes cannot see faults
  in observations or actions, which leave the episodes as the environment
  produced them.

A boundary fault is a difference between what the environment took and
gave and what the trainer sent and received. So:

**Does a step ledger kept on both sides of the boundary, compared step by
step, catch every fault in what crosses, and locate it? Does it do so with
no reference run, and with the two sides on different machines?**

---

## Declared in advance: what was already known

1. **The code**, E54's with two additions.
   - `faulty_client.py`: E54's client. Each environment is wrapped in
     `LedgerEnv`, a gymnasium wrapper outside the client's own code. It
     records, for each step:
     - the observation the environment had given before it;
     - the action it was handed;
     - the reward, flags and observation it returned.
     `--ledger` saves the record.
   - `train.py`: E54's, plus two things:
     - it passes `--ledger` to the client;
     - `--external-client`: the bridge listens on every address and starts
       no client, for runs whose client is on another machine (`cross.sh`).
     Its `--trace` keeps the trainer's side, as in E54.
   - `ledger.py` compares the two sides, env by env, step by step:
     observation, action, reward, episode end, time limit, final
     observation, and the number of steps. It needs no fault-free run.
   - `jobs.py`, `run.sh` (same machine, ports 8870-8879), `cross.sh`
     (across machines, port 8880), and E54's `summarise.py`.
2. **The faults and doses** are E54's (its `PROTOCOL.md`), at doses one,
   0.01 and 1.0.
3. **The machines.**
   - guangzhao: E54's `bridges-venv`.
   - the laptop: a Windows 11 laptop, Python 3.11 with gymnasium 1.3.0 and
     numpy 2.3.2. It reaches guangzhao over Tailscale at about 5 ms.
4. **The pilots**, at seed 9, in `results/pilot/`.
   - **Same machine** (`same-ledger.csv`, `same-summary.txt`): 72 runs.
     - `none` ended on E50's pilot weights (`3721266c`), so the wrapper
       changes nothing that crosses.
     - The ledger caught all 63 client-fault runs with a changed value, at
       their first changed cell and field.
     - It found no difference in the 3 controls and the 6 log-fault runs:
       72 of 72 right.
     - One defect, fixed before this protocol: `ledger.py` crashed on a run
       whose envs never stepped (`step:fill` at every value).
   - **Across machines** (`cross-ledger.csv`): 7 runs, at one cell where
     there was a fault:
     - `none`, `obs:stale`, `action:zero`, `reward:stale`,
       `truncated:as-term`, `step:fill` and `log:short`;
     - 7 of 7 right.
     - The fault-free run's weights were `be73067e`, not guangzhao's
       `3721266c`. Across machines, byte identity would have called a
       fault-free run faulty; the ledger found no difference.

---

## Design

- **Same machine** (`bash run.sh results registered 10` on guangzhao):
  Pendulum, seeds 0-2. The 3 controls, plus the 21 client faults and 2 log
  faults at doses one, 0.01 and 1.0. Every run is traced and ledgered:
  3 x (3 + 69) = 216 runs.
- **Across machines** (`bash cross.sh results/cross 0` on the laptop):
  Pendulum, seed 0. SB3 and the bridge run on guangzhao, the env client on
  the laptop. `none`, then the 21 client faults and 2 log faults at one
  cell: 24 runs, one at a time.

Then `python ledger.py results` and `python ledger.py results/cross`, and
`python summarise.py` on each, for E54's checks.

---

## How the ledger is judged

A run is **caught** when anything differs between the two sides.

Its first difference is the lowest step that differs, then the lowest env,
then the first field in the order above. It is **right** when:
- for a client fault that changed a value, that step and env are the
  fault's first changed cell (`fault.json`), and the field is the one the
  fault touches (`ledger.py` maps each);
- for a control, a log fault, or a run whose fault changed no value,
  nothing differs.

---

## Checks that the runs are valid

- **V1:** every run reaches its 102,400 steps, and its client exits 0.
- **V2:** on guangzhao, `none` ends on E50's weights at seeds 0, 1 and 2
  (`ec5b8226`, `062e92e3`, `3a8d968f`): the ledger's wrapper changes nothing
  that crosses.

---

## Predictions, and what falsifies each

**P1 - On one machine, the ledger catches and locates every fault in what
crosses.** Every client-fault run with a changed value is caught, at every
dose, and its first difference is right. Falsified by one run that is not
caught, or not right.

**P2 - It raises no false alarm.** No control and no log-fault run shows a
difference, on one machine or across machines. Falsified by one that does.

**P3 - Across machines too.** With the env client on the laptop, all 21
client faults at one cell are caught, and each first difference is right.
`none` shows no difference. Falsified by one run against either.

**P4 - Byte identity cannot judge across machines.** The fault-free run
across machines ends on weights other than guangzhao's fault-free run at
seed 0 (`ec5b8226`). Falsified if they are the same.

**P5 - The ledger and the record catch every silent fault, on any
machine.** Every client- and log-fault run with a changed value is caught
by the ledger or by E54's record, on one machine and across machines.
Falsified by one run that neither catches.

> Grounds: the pilots, above. P1-P3 follow from what a boundary fault is,
> if the ledger sits outside the client's code: the pilots test that it
> does. P4 is E43's and E50's finding that a seed belongs to a machine.

**Reported, not predicted:**
- the time each run took across machines;
- E54's checks on every run: rule, weights and record.
