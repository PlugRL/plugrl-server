# E54 measurement protocol (pre-registered)

**Written 2026-10-02, after the pilots in `results/pilot/`, before the
registered runs.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

Across machines and languages, where byte identity is impossible, E43, E44,
E48 and E49 judge a boundary by a status rule: the policy still learns. E51
and the failures in the paper's Table III show that faults at a boundary can
raise no error. So:

**When something goes wrong at the environment-trainer boundary, which
checks notice? For each field that crosses the wire and each way it can go
wrong, from one cell to every cell, does each of these catch it:**
- **the status rule** the paper uses across machines;
- **a strict curve check**: the final return against ten fault-free seeds;
- **byte identity**: the final weights against the fault-free run at the same
  seed, on the same machine;
- **reconciliation**: the trainer's log of episodes against a record kept
  where the environments produce the rewards;
- **the bridge's own log**, which counts on the trainer's side of the wire?

**And can the first difference from the fault-free run say where the fault
is?**

---

## Declared in advance: what was already known

1. **The code**, synced to guangzhao and hashed into each run's directory.
   - `train.py`: E50's bridge arm (SB3 2.9.0 PPO, 16 envs, one torch thread,
     CPU, VecMonitor, `PPO(seed=s)`), with `--fault`, `--dose` and `--trace`.
     Pendulum-v1 at rl-zoo's settings for 102,400 steps; HalfCheetah-v5 at
     SB3's defaults for 409,600 steps, both as in E50. The bridge's timeout is
     60 s, so a run whose client is gone stops at once.
   - `faulty_client.py`: E50's client, plus the faults, and a record of each
     episode's return and length as the environments produced them, before
     any fault.
   - `jobs.py`, `run.sh`: the runs, 10 at a time, each on its own port.
   - `localise.py`, `summarise.py`: the judgements below.
2. **The faults** (`faulty_client.py`'s docstring has each one).
   - **Client faults, 21.** Four fields with value operators:
     - `obs` (the infer observation), `action` (as the client applies it),
       `reward`, and `final-obs` (the observation an episode ended on);
     - operators: `zero`, `stale` (the step before's), `swap` (the next
       slot's), `f16` (rounded through float16), and for `action` also `clip`
       (to half its range).
     - Plus `terminated:spurious`, `truncated:drop`, `truncated:as-term` and
       `step:fill` (an env not stepped, its feedback filled in).
   - **Log faults, 2**, in the trainer's VecMonitor: `log:drop-first` (the
     return without the episode's first reward) and `log:short` (the length
     one short). E51 found both in Gymnasium 1.3.
   - **Loud faults, 2**, which the bridge rejects: `indices:reorder` and
     `reward:short-array`.
   - **Controls, 3**: `none`, `delay-1ms`, `wire-float64`.
3. **The doses.** `one` is a single cell, in env 0, the first eligible one at
   or after vector step 3,250 whose value the operator changes. `0.001`,
   `0.01`, `0.1`, `1.0` hit each eligible cell with that probability. The
   cells are drawn from a generator seeded 54, so every training seed gets the
   same ones. Each run counts the cells its fault changed.
4. **The pilots**, at seed 9, in `results/pilot/`. There were three.
   - **v1** (`results/pilot/v1.txt`): 13 hand-picked faults, each at every
     cell.
     - The fault-free run ended on E50's pilot weights (`3721266c`), as did
       `delay-1ms` and `wire-float64`. Every other fault changed the
       weights.
     - The status rule caught one fault of ten (slot-shifted rewards).
     - The fault-free run was the worst of the thirteen, at -680 against
       about -200 for most faults: seed 9 is an unlucky seed.
     - This pilot led to the design here: every field with every operator,
       at five doses. Its client and `train.py` were replaced by this
       design's; only its summary is kept.
   - **v2** (`results/pilot/v2.txt`): this design at seed 9. Two flaws, fixed
     before v3:
     - The one-cell dose sat on an episode's last step. That turned
       `terminated:spurious` into a time limit sent as a termination.
     - The localiser looked at messages before envs.
     - Also changed for v3: a one-cell dose moves on to the next cell when
       the operator would not change the value (`action:clip` within range
       had changed nothing).
   - **v3** (`results/pilot/v3.txt`) ran this protocol's code, except the
     strict check's false-alarm count, which was added to `summarise.py`
     after v3 started. In it:
     - **weights** caught every client-fault run at every dose (105 of 105),
       and no control or log-fault run;
     - **rule** caught 0, 0, 1, 1 and 9 of the 21 client faults at doses
       one, 0.001, 0.01, 0.1 and 1.0. Two of those it caught looked better
       than the fault-free run, not worse: with every reward zeroed, the log
       showed a perfect return of 0, and spurious terminations made short
       episodes with small returns;
     - **record** caught 4, 6, 6, 6 and 5 of the 21, and both log faults at
       every dose with a changed cell. It missed every `reward:f16`, and
       `reward:zero` and `reward:stale` at one cell, whose change to a
       return is under its tolerance;
     - **bridge** caught the log faults and nothing else;
     - weights or record caught every run with a changed cell;
     - both loud faults raised at once (ClientLost, once the bridge closed
       the connection);
     - localisation was right in 117 of 117 traced runs;
     - HalfCheetah at seed 9:
       - `none` ended on E50's pilot weights (`0260ca9d`);
       - `reward:stale` at one cell and `obs:f16` at 0.01 changed the
         weights and passed the rule;
       - record caught the first.
   - The strict check could not be judged in the pilots: there was no band.
     The seeds' final returns are not normal: E50's seeds 0-2 ended near
     -200, and seed 9 at -680.

---

## Design

On guangzhao, `bash run.sh results registered`: `jobs.py registered`, 10 runs
at a time, ports 8854-8863.
- **Pendulum, seeds 0, 1, 2:** the 3 controls; the 23 client and log faults
  at the 5 doses; the 2 loud faults at one cell. Every run traced. That is
  3 x (3 + 115 + 2) = 360 runs.
- **Pendulum, seeds 10-19:** fault-free, for the strict check's band. 10
  runs.
- **HalfCheetah, seeds 0, 1, 2:** `none`; the 23 client and log faults at
  doses `one`, `0.01` and `1.0`. Not traced. 3 x (1 + 69) = 210 runs.
- **HalfCheetah, seeds 10-19:** fault-free. 10 runs.

590 runs in all. Then `python localise.py results` and `python summarise.py
results`.

---

## How each check is judged

Batches are 16 consecutive episodes of a log. A check **catches** a run's
fault when:
- **raises:** the run stopped on an error.
- **rule:** the trainer's log fails the env's status rule, the one E44 (for
  Pendulum) and E43 and E48 (for HalfCheetah) used:
  - Pendulum: batches 20 to the last at least 500 above batches 1-2.
  - HalfCheetah: batches 21-25 at least 200 above batch 1.
  - A log too short to have those batches fails.
- **strict:** the mean of those last batches, in the trainer's log, is more
  than 3 standard deviations from the mean over the fault-free runs at seeds
  10-19.
- **weights:** the final weights' SHA-256 differs from the fault-free run's
  at the same seed.
- **record:** the trainer's log and the client's record differ in the number
  of episodes, or in an episode's length, or in its return by more than
  1e-4 relative plus 0.01 (E49's tolerance). Episodes are compared in the
  order they ended, then by env.
- **bridge:** the same comparison, between the trainer's log and the
  bridge's.

A run whose fault changed no cell is reported and counted as neither caught
nor missed.

**Localisation** (`localise.py`) is judged on every traced run. On one
machine the fault-free run at a seed is deterministic, so a faulted run
crosses the same bytes up to the fault. The localiser finds three things:
- the first vector step at which anything the trainer sent or received
  differs;
- the lowest env that differs at that step;
- the first of that env's messages that differs.

It is **right** when:
- that step and env are the fault's first changed cell;
- the message is the one the fault touches (`localise.py` maps each);
- for a control, a log fault, or a run whose fault changed no cell, the
  wire is identical.

---

## Checks that the runs are valid

- **V1:** every run but the loud faults' reaches its steps, with its client
  exiting 0.
- **V2:** `none` ends on E50's weights at seeds 0, 1 and 2, for both envs
  (`ec5b8226`, `062e92e3`, `3a8d968f`; `8261c03a`, `86564694`, `65cc8843`).
  This holds with 10 runs at a time on the machine.

A run that fails V1 is a defect to find, not a finding about a check.

---

## Predictions, and what falsifies each

**P1 - Byte identity sees every change.** Every client-fault run whose
fault changed at least one cell ends on weights that differ from the
fault-free run's at its seed. This holds on both envs, at every dose.
Falsified by one such run on the fault-free weights.

**P2 - and nothing else.** Every `delay-1ms`, `wire-float64` and log-fault
run ends on the fault-free run's weights at its seed. Falsified by one that
does not.

**P3 - The status rule misses most faults.** On Pendulum, client- and
log-fault runs with a changed cell:
- at doses one, 0.001, 0.01 and 0.1, the rule catches at most 10% at each
  dose;
- at dose 1.0, at most 60%.
Falsified by more at any dose.

**P4 - Reconciliation sees what changes the books, and only that.**
- It catches every run of these, at every dose with a changed cell except
  one cell:
  - `reward:zero`, `reward:stale`, `reward:swap`;
  - `terminated:spurious`, `truncated:drop`.
- It catches `step:fill` at doses 0.001 to 0.1, and both log faults at every
  dose with a changed cell.
- It catches no run of any `obs`, `action` or `final-obs` fault, nor of
  `truncated:as-term`.
- Falsified by one run against any of these.
- Reported, not predicted:
  - `reward:f16` at every dose;
  - the reward faults at one cell;
  - `step:fill` at 1.0, where no env ever steps, so no episode is logged on
    either side.

**P5 - The bridge's log sees only the trainer's side.** It disagrees with the
trainer's log in every log-fault run with a changed cell, and in no
client-fault run. Falsified by one run against either.

**P6 - Byte identity or reconciliation catches every silent fault.** Every
client- and log-fault run with a changed cell is caught by weights or by
record, on both envs. Falsified by one that neither catches.

**P7 - The bridge rejects the loud faults.** Both raise at every seed, and
end in under 120 s. Falsified by one that does not.

**P8 - The first difference localises the fault.** Localisation is right in
every traced run. Falsified by one wrong.

**P9 - HalfCheetah too.** The status rule catches at most 10% of client- and
log-fault runs with a changed cell at doses one and 0.01. Falsified by more
at either dose.

> Grounds: v3, above. The margins of P3 and P9 are wide because v3 had one
> seed. The exclusions in P4 follow from what the record compares, returns
> and lengths. A fault that changes neither is invisible to it by
> construction: which flag ends an episode, for example.

**Reported, not predicted:**
- the strict check at every dose, for both envs;
- its false alarms: each fault-free seed in 10-19 judged against the band
  of the other nine;
- every run's final return, cells changed and wall clock;
- HalfCheetah at dose 1.0, by every check.
