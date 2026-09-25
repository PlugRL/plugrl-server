# E23 amendments

Dated changes and declared deviations. `PROTOCOL.md` is not edited after the
first registered data point; anything learned afterwards goes here.

---

## 1. The first attempt was killed from outside, and all three seeds restart

**2026-09-25 16:05, before any of the first attempt's returns were read.**
This is declared deviation 1: one restart of any run that dies for a reason
outside the experiment.

The first attempt started at 15:42:57. All three seeds ran cleanly until
81,920 steps - 20 of their 100 iterations - and each saved its checkpoint
there. No server log holds a traceback. Between 15:53:12 and 15:53:57 every
log stopped at once, mid-iteration, and `run.sh`'s own output ends in four
failed forks:

```
dofork: child -1 - forked process ... died unexpectedly, exit code 0xC000026B
run.sh: fork: retry: Resource temporarily unavailable
```

`0xC000026B` is `STATUS_DLL_INIT_FAILED`: a new process that could not
initialise because the session it belonged to was being torn down. The time
matches the end of the Claude Code session that had launched `run.sh` as a
background task. That session's shell went, and it took the three servers,
the three clients and `run.sh` with it. Nothing in the experiment did this.

**What was done.** The first attempt's logs and checkpoints were moved,
untouched, to `results/attempt-1/`, and its returns were not read. All three
seeds restart from scratch with the same `run.sh`, the same code and the same
seeds, launched in a console of its own (`Start-Process`) so that the end of
a session cannot reach it. Every verdict reads the second attempt only.

**What it does to P3.** The wall clock is measured by the second attempt's own
`start` and `end`, so the lost ten minutes do not count against it.

**Addendum, 16:10.** The restart itself took three launches. The first used
`bash.exe` from the PATH, which on this machine is WSL's: it started one
seed-0 server under WSL paths and then lost its script; that server was
stopped before any client connected, so it collected nothing. The second
passed the command to `Start-Process` as an argument list, which does not
quote its elements, so `bash -c` received only `cd` and did nothing. The
third ran a launcher script with Git Bash and is the run every verdict reads:
`start 2026-09-25 16:08:14`, all three servers listening by 16:09:49. None of
the three touched `run.sh`, the code, or the seeds.

---

## 2. The machine was shared from 16:02, and P3 is not read

**2026-09-25 16:10, while the runs were starting and before any result.**
This is declared deviation 2.

From 16:02:40, and more heavily from 16:07:47, processes from a separate
Claude Code session on another project (`memeseeks`) were running on this
machine: a `memeseeks serve` on port 8766 and a UI-test script beside it. One
of them had used about 294 CPU-seconds within two minutes of starting. They
are not this experiment's and were left alone. Starting the three servers took
50 to 95 seconds, against 7 in the first attempt.

So P3, the wall clock, does not measure E23's cost and is **not read**. P1 and
P2 are read as registered. Learning here is not deterministic run to run
(E20), so a shared machine can change what is computed as well as how long it
takes, but P2's threshold is far enough from the null (14 to 19 against 500)
that this cannot decide it.
