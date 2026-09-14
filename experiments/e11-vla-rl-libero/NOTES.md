# E11 notes

Dated records of what happened while running this experiment. `PROTOCOL.md` is not edited. Anything that bears on how its results should be read is written here instead, with its date.

---

## 2026-09-13 - The first inference against a real pi0.5 crashed the server

The first smoke cell ran `Pi0Policy` through the real server loop for the first time: `libero_spatial`, one episode per task. It was not one of the protocol's cells. The server died on its first `infer`:

```
TypeError: Unsupported model observation type: <class 'list'>
```

openpi's `LiberoInputs` sets each `image_mask` entry to `np.True_` or `np.False_`, which is a numpy scalar, not an ndarray. PlugRL's `batch_aggregate` stacked only ndarrays, so each mask became a Python list, and the tensor conversion rejected it. `text` came through intact at every stage.

It is fixed in plugrl-server #14, commit `8b6aa27`, with a regression test. **Every E11 number comes from a server that includes that fix.** Each cell writes a per-file sha256 manifest of the source it ran to `results/environment-source-<cell>.txt`, with carriage returns stripped so each hash can be compared with the git blob.

**How the stopping rule was read.** `PROTOCOL.md` stops the experiment "if Stage A cannot run end to end". The crash happened in a harness check run before any protocol cell existed, so it is recorded here rather than treated as a Stage A result.

## 2026-09-13 - Workers from a killed run joined the next server

The second smoke cell reported 10 episodes from its clients, while the server recorded 20.

**The extra ten came from the first smoke cell.** After that cell's server crashed, its env clients were killed by matching the CLI's name. LIBERO's multiprocessing workers run as `python -c from multiprocessing.spawn ...`, so they survived. An env client retries while its server is away, and these kept retrying for about half an hour. They joined the next server the moment it listened, ran one episode per task, and exited. Their ten `env_steps` match the ten extra episode lengths exactly.

**Only the server's own record could show this.** The client-side rows of the cell were clean. The harness's reconciliation said "false", but for a different reason: it compared the server's step counter, which the eval algorithm leaves at 0, so it could not have been true for any eval cell.

**A second leak.** The same cell also left its server running after teardown, because the harness killed the subshell rather than the server. The next launch found it still holding GPU 0.

**Changes made before any protocol cell ran:**

- **The harness now refuses to start** while any LIBERO client or pi0 server process is alive. It runs the server and the clients each in their own session, tears down whole process groups, stops the clients if the server dies, and checks that nothing survived.
- **Reconciliation compares the right quantities.** It compares the server's episode count with the clients', and the sum of the server's episode lengths with the clients' environment steps. Replayed on the contaminated output, it reports 20 against 10 and 2244 against 1139, and marks every row invalid.
- **`valid` now includes the accounting check.** `PROTOCOL.md` says a row whose step accounting disagrees is invalid. The first version of the harness did not implement that; this one does.

**A third smoke cell ran on 2026-09-14 under the hardened harness.** It completed 10/10 episodes with client exit 0. The server recorded 10 episodes and 1052 steps against the clients' 10 and 1052. There were zero missing-step-state warnings and zero reconnects, and teardown was clean. Smoke rows are written outside `results/` and are not part of any statistic.

## 2026-09-14 - LIBERO renders on a different GPU than the harness names

The harness gives the env clients `CUDA_VISIBLE_DEVICES=1` and `MUJOCO_EGL_DEVICE_ID=1`. During the third smoke cell, `nvidia-smi` showed the ten clients' EGL contexts on GPU 2, at 526 MiB each, not on GPU 1. EGL enumerates devices in a different order from CUDA.

All eight GPUs on the node are the same model, so this does not change anything measured. It matters only for knowing what else shares that GPU.
