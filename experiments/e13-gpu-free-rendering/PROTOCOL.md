# E13 measurement protocol (pre-registered)

**Written 2026-09-18, before the measurement it describes was run.** Several
numbers were already in hand and are declared below rather than dressed as
predictions.

This file must not be edited after the first registered data point.

---

## The question

E12 showed the environment side can be installed without CUDA: 3.4G and no
`nvidia-*` wheels for LIBERO, sending byte-identical observations. It left one
GPU dependency untouched, and said so: **rendering**. LIBERO renders through
EGL, which comes from the system driver rather than from a Python package, so
every run so far has drawn frames on an NVIDIA card.

A rollout machine with no GPU has to render in software. The question is not
whether it can - it can, and that is declared below - but **what it costs in
the loop the project actually runs**.

That distinction matters because a single client and ten clients are different
regimes. With one client the environment waits on the policy; with ten sharing
one server, the server's model lock serialises inference and each client waits
on the queue instead. Rendering that is 11x slower may be decisive in the
first regime and invisible in the second.

---

## Declared in advance: what was already known

Exploratory measurements taken before this protocol was written:

1. **robosuite's EGL backend cannot do software rendering.**
   `robosuite/renderers/context/egl_context.py` requires the
   `EGL_PLATFORM_DEVICE_EXT` extension, which Mesa does not provide; forcing
   glvnd to Mesa raises `ImportError: Cannot initialize a EGL device display`.
   A bare `mujoco.Renderer` does succeed on Mesa, which is why the first
   attempt looked like it worked. The usable path is `MUJOCO_GL=osmesa`,
   which robosuite supports through `osmesa_context.py`.
2. **`libosmesa6` is not installed on the cluster.** It was extracted from the
   Ubuntu 22.04 package into `$R/opt/osmesa` and reached through
   `LD_LIBRARY_PATH`; the shared container was not modified.
3. **Stepping cost, one client, no server**: 39.6 steps/s with NVIDIA EGL,
   **3.59 steps/s** with osmesa - a factor of 11.
4. **One client in the loop**: 3 of 3 successes either way. Collection took
   36.2 s with EGL and **111.2 s** with osmesa, a factor of 3.07. The share of
   the loop spent waiting for the policy fell from 0.787 to 0.217, and the
   share spent stepping rose from 0.235 to 0.782.

None of that is the registered result. All of it is single-client, and the
question above is about ten.

---

## Held fixed

- **Server**: unchanged, `pi0-policy` / `pi05_libero` / `eval`, CUDA, GPU 0.
  It is the same process in both cells; only the clients differ.
- **Client install**: E12's CUDA-free environment, `venv-libero-cpu`, in both
  cells. The install is not a variable here; the renderer is.
- **Task**: `libero_spatial` task 0, initial states in order, seed 7,
  `replan-steps 5`, one environment per process.
- **Episodes**: 3 per client process, so ten processes produce 30.
- **Machine**: cluster `qz103`, 128 cores, load under 50 at the start of each
  cell. Software rendering is CPU work and ten of them run at once; the load
  at start is recorded with each cell.

## The measurement

Two cells, run back to back, ten client processes each against one server:

| cell | renderer | GPU used by the client |
|---|---|---|
| `ten-egl` | NVIDIA EGL | yes, for rendering |
| `ten-osmesa` | Mesa osmesa, software | none |

Recorded per cell: wall clock from the first client starting to the last
exiting, and from each process's `summary.json` the episodes, successes, env
steps, `collect_time_s`, `env_step_s`, `infer_wait_s` and the fractions.

## Predictions, recorded before the data

1. **Both cells complete, 30 episodes each, no client failing.** Falsified by
   any client exiting non-zero or producing fewer than 3 episodes.
2. **`ten-osmesa` wall clock is less than 1.3x `ten-egl`'s** - that is, the
   11x stepping penalty is mostly hidden by the queue. Falsified at 1.3x or
   above. This is the prediction the experiment exists to test; the
   single-client factor was 3.07x.
3. **In `ten-osmesa`, `env_step_frac` is below 0.5** for the median client:
   stepping is no longer the dominant term once ten clients share one policy.
4. **Success counts are within 3 of each other** across the two cells, out of
   30. Not a statistical claim, a sanity check: the renderer should not change
   what the policy achieves, and with an unseeded policy exact equality is not
   available.

## Known limitations, stated in advance

- **One task, three episodes per client.** This measures the loop's timing,
  not a success rate.
- **The server is on the same machine.** Network latency is E7's question,
  measured there at +0.52 ms; it is not re-measured here, so these numbers
  describe a co-located split.
- **Software rendering is CPU work on a 128-core machine.** A robot's onboard
  computer has far fewer cores, and this says nothing about how ten renderers
  would fare on four.
- **Rendering was not verified pixel for pixel.** Software and hardware
  rasterisers need not produce identical frames. The comparison here is of
  timing and task success, not of images.
- **The unseeded policy** means step counts differ between any two runs. E12
  established this; it is why prediction 4 is a range.

## Stopping rules

- Each cell runs once. A cell that fails is recorded as failed, and the cause
  is investigated in the findings rather than by rerunning until it passes.
- If `ten-egl` cannot be run because the GPU is busy with someone else's work,
  the experiment waits rather than comparing against a number from a
  differently loaded machine.

## Record format

- `results/summary.tsv`: one row per cell
- `results/<cell>/`: the per-process `summary.json` files as produced
- `results/verify_render.sh`, `results/render_bench.py`: the harness, as run
- `FINDINGS.md`: what was asked, what came back, what it does not support
