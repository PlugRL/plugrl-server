# E48 amendment

**2026-10-02 00:55, during the cross arm. Cross seed 0 had finished on
guangzhao; its returns had not been looked at.**

## What happened

`run_e48.sh` hands `cross_clients.py` the run's directory on guangzhao,
`/home/guangzhao/zuogou/plugrl/e48/cross-seed<s>`. Git Bash converts an
argument that looks like a POSIX path when it starts a Windows program, so
the script received `D:/Program Files/Git/home/guangzhao/...`. Its poll then
ran `grep -q RUN_DONE` on a path that does not exist on guangzhao, every 10 s,
and never saw the run end.

Meanwhile RLinf's run of cross seed 0 ended at 00:26:24 with rc 0. RLinf
does not close its environments when it ends: the bridge's connection dropped
with no close frame, so the bridge never sent `plugrl-server-stop`. Following
SPEC.md, the laptop's 16 clients took the drop for a lost connection and
kept reconnecting, 3,190 attempts by 00:51. Seed 1 could not start, since
`run_e48.sh` waits for `cross_clients.py`.

## What was done

`tmp/e48_stop_watcher.sh` (copied to `results/stop-watcher.sh`) does what the
poll was meant to do. For each cross seed it waits for RUN_DONE in that run's
`run.txt` on guangzhao, then stops that seed's client tree on the laptop.
`cross_clients.py` then returns as designed, and `run_e48.sh` carries on.
Its log is `results/stop-watcher.txt`. Seed 0's clients were stopped at
00:53:14, and seed 1 started at 00:53:18.

`run_e48.sh` and `cross_clients.py` were not edited while they ran.

## What it changes

- **Nothing that is judged.** Each RLinf run had finished before its clients
  were stopped. The watcher only ends clients that are idle and reconnecting.
  V1, V2, P1 and P2 read RLinf's log and TensorBoard events, which are
  complete by then.
- **The wall clock between seeds.** It includes seed 0's 27 idle minutes,
  which are reported as such.
- **`clients exit` in `e48.txt` is 1 for watcher-stopped seeds**, because
  `taskkill /F` ends the tree. The clients' own log shows the reconnecting
  that preceded it.

## Found, for later

1. `cross_clients.py` should get the remote path in a form Git Bash leaves
   alone: `MSYS_NO_PATHCONV=1`, or a path it does not convert.
2. A trainer that ends without closing its environments leaves remote clients
   reconnecting forever. The bridge could send stop from an exit handler,
   though that does not help when the process is killed. Alternatively the
   env client could give up after a bounded time with no server.
