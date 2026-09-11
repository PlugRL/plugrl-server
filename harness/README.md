# plugrl-sweep

Runs the PlugRL evaluation matrix and turns what comes back into the two
tables the paper needs: which (environment, policy) combinations run, and
where their wall clock goes.

```bash
python -m plugrl_sweep.sweep \
    --envs dummy-v1 classic-v1 libero-v1 \
    --policies dummy-policy \
    --seeds 0 1 2 \
    --output-dir sweep-out \
    --policy-override discrete=false \
    --policy-override action_dim=7 \
    --env-policy-override classic-v1:discrete=true \
    --env-policy-override classic-v1:action_dim=2
```

```
| env        | dummy-policy |
|------------|--------------|
| atari-v1   | -            |
| classic-v1 | ok           |
| dummy-v1   | ok           |
| libero-v1  | -            |

4 combinations: ok=2, -=2

env              policy              n      fps     env   infer    pack      fb
-------------------------------------------------------------------------------
classic-v1       dummy-policy        2      9.4    2.3%   96.2%    0.2%    1.3%
dummy-v1         dummy-policy        2     37.2    3.3%   96.4%    0.1%    0.3%
```

## What it is careful about

**Telling "we did not try" from "we tried and it failed."** A matrix that
blurs those is worse than no matrix - a reader cannot tell a broken
combination from an uninstalled one. Three signals separate them:

| Signal | Verdict |
|---|---|
| tyro rejects the uid (`invalid choice: 'libero-v1'`) | `unavailable` - the extra never registered the env |
| traceback ends in `DependencyNotInstalled` or an `ImportError` naming an extra | `unavailable` - registered, but the machine lacks something |
| anything else that started and failed | `fail` - a real result about the combination |

Getting this wrong is easy and was got wrong twice while building this. The
env client warns about *every* extra it could not import, on *every* run, so
a log always contains "d4rl is not installed" no matter what was asked for;
matching on that phrase marked the entire matrix unavailable. And tracebacks
print qualified names, so `gymnasium.error.DependencyNotInstalled` does not
equal `DependencyNotInstalled`. Both are regression-tested.

**Not believing an exit code.** A cell is `ok` only if the client's own
`summary.json` says it completed the episodes it was asked for. An earlier
hand-written harness in this project reported 118879 exchanges/s from a
client that had died on connect; a run that exits zero without finishing is
`incomplete`, not success.

**One failed seed sinks the combination.** Two of three seeds working is not
a working combination, so `combine_seeds` takes the worst outcome. The
exception is `unavailable`, which only survives if every seed was
unavailable - if some seeds ran, "not installed" is not the story.

**Only completed runs are timed.** A failed run's partial timings describe
how far it got, not how the combination behaves.

## Per-environment policy configuration

A policy has to match the environment's action space, and environments
disagree: CartPole takes one discrete action, LIBERO takes a seven-dimensional
continuous one. `--policy-override` sets the default; `--env-policy-override
ENV:KEY=VALUE` layers on top for the envs that differ. Without this the
matrix can only ever be one column wide.

The distinction this buys is worth stating: a cell that fails because the
policy does not fit the action space is a fact about *that pairing*, not
about the environment. `classic-v1` fails against a 7-dim continuous policy
and passes against a discrete 2-dim one, and the matrix should say so.

## Resuming

Results are appended to `results.jsonl` as each cell finishes, and a rerun
skips cells already present. On a cluster this matters more than it sounds:
a sweep that must complete or be thrown away will be thrown away.

Pass `--no-resume` to start over.

## Layout

| File | |
|---|---|
| `cell.py` | one matrix cell, and the two command lines it becomes |
| `outcome.py` | the verdict types and the classifier |
| `runner.py` | runs one cell: server up, client through, server down |
| `sweep.py` | drives the matrix, appends results, prints the tables |
| `aggregate.py` | folds seeds, renders the matrix and timing tables |

`scripts/example-sweep.sh` runs a matrix against real executables that
deliberately contains a cell of each kind. A run that does not produce three
different verdicts means the harness has stopped being able to tell them
apart, which is the one thing it exists to do. It takes the two binaries from
`PLUGRL_SERVER_BIN` and `PLUGRL_CLIENT_BIN`, because the server needs torch
and the env client deliberately does not - they usually live in separate
environments.

## Not done yet

- **Cluster submission.** Everything here runs locally. A Slurm or PBS
  backend would replace the `subprocess` calls in `runner.py` and nothing
  else; the matrix, classification and aggregation are scheduler-agnostic.
- **Parallel cells.** One at a time. Ports are already allocated per cell,
  so the constraint is GPU contention on the server side rather than the
  harness.
- **Server-side metrics.** Only the client's `summary.json` is collected.
  The server's own metrics (learn time, inference batch sizes) are written to
  its metric sink and not yet joined in.
