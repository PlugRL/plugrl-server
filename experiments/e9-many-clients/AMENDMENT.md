# E9 protocol amendment 1

**Written 2026-09-11, after an audit of the pre-registration and before any
E9 data exists.**

`PROTOCOL.md` says it must not be edited, and that if a rule turns out to be
flawed the fix is a new document saying why the old one was unusable, with
both kept. E9 has not run, so the letter of that rule does not bind yet - but
a pre-registration whose text can be quietly corrected after publication is
not a pre-registration. **`PROTOCOL.md` is left exactly as written.** Two
sentences in it are wrong. Both are wrong in their citations rather than in
their substance, and both are corrected here.

Citations below name sections and quote text rather than line numbers, since
the sibling findings files are being edited while this is written.

---

## Correction 1: only E7 states the one-client limit

`PROTOCOL.md` line 16 says:

> E5, E7 and E8 all say so under their limitations.

**Only E7 does.** What was checked, in the copies under
`plugrl-server/experiments/` on 2026-09-11:

* `e7-cross-machine/FINDINGS.md`, under "What this does and does not
  support": "Nothing about many clients at once. Every run here is one
  client." Correct.
* `e5-boundary-cost/FINDINGS.md` has no limitations section containing it.
  Its headings are "In one sentence", "Payload size sweep", "Splitting the
  fixed overhead", "Two conclusions", "One configuration that could not be
  measured" and "Reproducing". The only plural "clients" in the file is the
  bandwidth projection "a gigabit link saturates at roughly 22 clients",
  which is a planning figure, not a limitation.
* `e8-keepalive-hypothesis/FINDINGS.md` has a "Not supported" block with
  three bullets: what trips the keepalive in normal operation, frequency,
  other hardware. The word "clients" does not appear anywhere in that file
  (`grep -c clients` returns 0).

**The underlying fact is true**, and was checked separately: E5 invokes one
C++ client binary per measurement, one at a time
(`e5-boundary-cost/run.sh:74`), and E8 launches a single client process
(`e8-keepalive-hypothesis/run.sh:60`) without passing `--num-procs`, whose
default is 1 (`plugrl-env-client/src/plugrl_env_client/cli.py:25`).

**The correct sentence is:** E7 says so under its limitations. E5 and E8 each
ran one client and did not state it as a limit.

## Correction 2: the `num_envs=1` constraint is not established in E8

`PROTOCOL.md` lines 49-50 say:

> one process per client, since `MuJoCoEnv` supports `num_envs=1` only -
> established in E8 and not rediscovered here.

**The constraint is real; E8 does not establish it.** The E8 directory holds
`FINDINGS.md`, `run.sh` and `results/`. `run.sh:18` defaults
`NENV="${4:-1}"` and passes it through as `--num-envs`; that is a default,
not a finding. `num_envs`, `num-envs` and `MuJoCoEnv` do not occur in E8's
`FINDINGS.md` at all. A reader sent to E8 for this finds nothing.

The constraint lives in the client:

```
plugrl-env-client/src/plugrl_env_client/envs/mujoco/mujoco_env.py:66-70
    if self.num_envs != 1:
        raise ValueError(
            "MuJoCoEnv only supports num_envs=1; run more client processes "
            "instead, which is what --num-procs is for."
        )
```

**The correct citation is:** one process per client, since `MuJoCoEnv` raises
on `num_envs != 1`
(`plugrl-env-client/src/plugrl_env_client/envs/mujoco/mujoco_env.py`) and
directs you to `--num-procs`.

## What this changes about the experiment

Nothing. Both errors are attributions. The held-fixed rule - one process per
client - stands unchanged, and so do the independent variables, the metrics,
P1 to P4, the stopping rule and the record format. What was wrong was where a
reader is sent to check them.
