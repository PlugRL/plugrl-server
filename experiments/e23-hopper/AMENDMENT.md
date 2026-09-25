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
