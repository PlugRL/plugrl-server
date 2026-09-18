# E12 measurement protocol (pre-registered)

**Written 2026-09-18, before the measurement it describes was run.** One
number was already in hand when this was written, and it is declared below
rather than presented as a prediction.

This file must not be edited after the first registered data point. If a rule
turns out to be unusable, write a new version saying why and keep both.

---

## The question

E1 round 4 measured what a machine that only produces rollouts has to carry:

| Case | Role | Packages | Disk | CUDA |
|---|---|---|---|---|
| envclient-bare | rollout | 30 | 224M | no |
| envclient-d4rl | rollout | 69 | 1008M | no |
| **envclient-robomimic** | rollout | 111 | **7.2G** | **yes (16 wheels)** |

and concluded, in `e1-dependency-conflict/FINDINGS.md`:

> `envclient-robomimic` drags in `torch==2.14.0` and 16 nvidia wheels by
> itself, because robomimic is a deep learning library that ships its own
> policy learning code. For environments like it, "the environment side needs
> no GPU" is simply not true.

That is true of a **default install**. On Linux, `pip install torch` gives a
CUDA build whether or not CUDA will be used. The environment side of PlugRL
steps a simulator and renders frames; it runs no model and holds no weights.

**Does the CUDA half come along because the environment needs it, or only
because it is what torch ships by default?**

This matters for one concrete thing, the same thing E1 named: whether a
robot's onboard computer, a CI runner or a laptop can be a rollout source for
a robomimic-class environment, including LIBERO, which E1 never measured.

### What this cannot settle

Nothing about speed. A CPU build of torch is not being asked to compute
anything here; if it were, this would be a different experiment. Nothing about
whether the environment *renders* without a GPU either - that is EGL and the
system driver, not a Python package, and it is measured separately in E13 if
it is measured at all.

---

## Declared in advance: what was already known when this was written

Before this protocol existed, one exploratory measurement was taken. A copy of
E11's `venv-libero` had its CUDA torch replaced with the CPU build of the same
version, and the orphaned `nvidia-*` and `triton` packages removed:

- 6.8G before, **2.4G** after, 15 nvidia wheels and triton gone
- `torch.version.cuda` is `None` and `torch.cuda.is_available()` is `False`

That is a modified copy, not an install, and it is not the registered result.
It is recorded here because it was known before the predictions below were
written, and predicting an outcome one has already seen is not a prediction.

---

## Held fixed

- **Method**: E1's `run-footprint.sh` - `uv venv`, then one `uv pip install`,
  then count packages with `uv pip list`, disk with `du -sb`, and CUDA by
  whether any `nvidia-*` or `triton` package is present. The same predicate,
  so the rows can sit in the same table.
- **uv 0.9.9**, the version E1 used.
- **Python 3.11**, as E1 used.
- **Package sets**: E1's `ENVCLIENT` and `ROBOMIMIC` lists, verbatim.
- **The CPU pin**: `torch==2.14.0+cpu` and `torchvision==0.29.0+cpu` - the CPU
  build of the exact version a default install resolves to, so that version
  is not a confounder.
- **Machine**: the cluster (`ssh qz103`), not E1's WSL2 laptop. GitHub is
  unreachable there, so `plugrl-protocol` and `libero` install from the local
  checkouts that E11's own environment installs from - the same source, by
  path instead of by URL. The absolute numbers may therefore differ from E1's;
  the comparison that matters is between rows measured here.

---

## The measurement

Four fresh installs, each built and discarded in turn:

| Case | Contents |
|---|---|
| `envclient-robomimic` | ENVCLIENT + ROBOMIMIC. E1's row, rebuilt here |
| `envclient-robomimic-cpu` | the same, plus the CPU torch pin |
| `envclient-libero` | ENVCLIENT + ROBOMIMIC + LIBERO |
| `envclient-libero-cpu` | the same, plus the CPU torch pin |

Recorded per case: packages, bytes, wall-clock install time, CUDA yes/no, and
whether `import robosuite` succeeds afterwards. **An install that is smaller
but cannot import is a failure, not a result.**

## The behaviour check

Size is not the claim; a working rollout source is. One evaluation run against
an unchanged pi0.5 server, with the CUDA-free client:

- `libero_spatial` task 0, 3 episodes, initial states in order, seed 7 - the
  same configuration a run with the CUDA client already produced
- **Passes only if** successes and env steps match that run exactly: 3 of 3,
  301 env steps, 61 inference calls, 61 feedback calls

Determinism is what makes this a check rather than an impression: same states,
same policy, same seed. A different number means the install changed the
environment's behaviour, not just its size.

---

## Predictions, recorded before the registered data

1. **`envclient-robomimic-cpu` reports CUDA: no.** Falsified if any
   `nvidia-*` or `triton` package survives a fresh install with the CPU pin.
2. **`envclient-libero-cpu` is under 3.5G.** A guess anchored on the
   exploratory copy, which is not the same construction; a fresh install
   resolves its own dependency set and may differ.
3. **Both CPU cases import `robosuite`.** Falsified by an import error.
4. **The behaviour check matches exactly.** Falsified by any difference in
   successes or step count.
5. **`envclient-libero` (GPU) exceeds E1's 7.2G robomimic row**, since LIBERO
   is robomimic plus its own assets.

If prediction 1 fails, the answer to the question is no, and E1's sentence
stands as written.

---

## Known limitations, stated in advance

- **A CPU torch is still torch.** This removes CUDA, not the deep learning
  library. The remaining install is hundreds of megabytes of a framework the
  environment never calls. Removing it would require patching LIBERO's imports
  and is not attempted here.
- **One machine, one Python, one moment.** Dependency resolution moves. These
  numbers describe what resolved on the day, which is why the log of each
  install is kept.
- **Disk is not the only cost.** RAM at runtime, first-import time and wheel
  download size are not measured.
- **The behaviour check is one task and three episodes**, chosen because a
  matching run already exists. It shows the client still works; it does not
  show it works for every environment in the suite.

## Stopping rules

- The four installs run once each. A case that fails is recorded as failed and
  is not retried with different pins; a retry would be a new protocol.
- The behaviour check runs once. If it fails, that is the result, and the
  cause is investigated in the findings rather than by rerunning until it
  passes.

## Record format

- `results/summary.tsv`: one row per case, tab separated, columns
  `case, packages, bytes, seconds, cuda, imports`
- `results/<case>.log`: the full install log for that case
- `results/behaviour-check.txt`: the run's summary, both counts, verbatim
- `FINDINGS.md`: what was asked, what came back, and what it does not support
