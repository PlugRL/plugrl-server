# E12: a rollout machine can drop CUDA after all

**E1 concluded that for robomimic-class environments, "the environment side
needs no GPU" is simply not true. That holds for a default install and not for
a deliberate one. Pinning the CPU build of the same torch takes the robomimic
env client from 7.2G with 16 nvidia wheels to 2.7G with none, and LIBERO from
7.8G to 3.4G, with `import robosuite` still working and the observations it
sends byte-identical to the CUDA client's.**

Pre-registered in [`PROTOCOL.md`](PROTOCOL.md) on 2026-09-18, before the
installs were run. One exploratory number was in hand first and the protocol
declares it rather than predicting it.

## What ran

| | |
|---|---|
| Machine | cluster `qz103`, Lustre working directory |
| Method | E1's `run-footprint.sh`: `uv venv`, one `uv pip install`, count with `uv pip list`, size with `du -sb`, CUDA by whether any `nvidia-*` or `triton` package is present |
| uv | 0.9.9, the version E1 used |
| Python | 3.11 |
| Package sets | E1's `ENVCLIENT` and `ROBOMIMIC` lists, verbatim |
| CPU pin | `torch==2.14.0+cpu`, `torchvision==0.29.0+cpu` - the CPU build of the version the default install resolves to |
| `plugrl-protocol`, `libero` | from the local checkouts, because GitHub is unreachable from the cluster. Editable installs in E11's own environment point at the same directories |
| Script | [`results/footprint_cluster.sh`](results/footprint_cluster.sh), as run |

## The measurement

| case | packages | disk | install | CUDA | imports |
|---|---|---|---|---|---|
| `envclient-robomimic` | 111 | **7.2G** | 71 s | **yes (16 wheels)** | ok |
| `envclient-robomimic-cpu` | 92 | **2.7G** | 68 s | **no** | ok |
| `envclient-libero` | 164 | **7.8G** | 79 s | **yes (16 wheels)** | ok |
| `envclient-libero-cpu` | 145 | **3.4G** | 61 s | **no** | ok |

Exact bytes are in [`results/summary.tsv`](results/summary.tsv); each install's
log is beside it.

**The first row reproduces E1 exactly.** E1 measured `envclient-robomimic` at
111 packages, 7.2G and 16 nvidia wheels on a WSL2 laptop; this run, on a
different machine some weeks later, resolves the same 111 packages to the same
7.2G with the same 16 wheels. The comparison between rows is therefore not
between two methods, and the CPU rows can sit in E1's table unchanged.

**19 packages carry the CUDA half**: 15 `nvidia-*` wheels, `triton`, and the
difference between the CUDA and CPU builds of torch and torchvision. For
robomimic that is 4.5G of 7.2G; for LIBERO, 4.4G of 7.8G.

## What it means for where a rollout can run

E1 asked whether a lab workstation, a CI runner or a robot's onboard computer
can be a rollout source. For robomimic-class environments its answer was no.

| environment | default install | with the CPU pin |
|---|---|---|
| robomimic | 7.2G, CUDA | **2.7G, no CUDA** |
| LIBERO | 7.8G, CUDA | **3.4G, no CUDA** |

The training server this project ships is 6.5G and wants a GPU. A 3.4G
rollout source that needs neither CUDA nor a GPU driver is a different class
of machine: it fits where the trainer does not.

This does not say the environment renders without a GPU. LIBERO renders
through EGL, which comes from the system driver rather than from a Python
package, and these runs still rendered on the cluster's cards. Whether
software rendering is usable is a separate question and is not measured here.

## Does the smaller client still behave the same?

[`results/behaviour-check.txt`](results/behaviour-check.txt) has the full
record. In short:

**The check as the protocol wrote it failed, and the protocol was wrong.** It
required an exact match against an existing run - 3 of 3 successes and 301 env
steps. The CUDA-free client returned 3 of 3 and **296**. Prediction 4 is
falsified as written.

The premise was the error. The protocol assumed the run is deterministic
because the environment is seeded. The policy is not seeded:
`openpi_policy.py:119` draws fresh noise for every inference, and nothing in
`src/plugrl_server` calls `manual_seed`, `np.random.seed` or
`torch.use_deterministic_algorithms`. A control run - the **CUDA** client
through the same harness - returned **300**, so the exact-match check could
not have passed for any client, including the unmodified one.

| run | client | success | env steps | infer |
|---|---|---|---|---|
| demo capture | CUDA | 3/3 | 301 | 61 |
| control | CUDA | 3/3 | 300 | 61 |
| under test | **CPU** | 3/3 | **296** | 60 |

**A check that does isolate the client passes.** An episode's first
observation depends on the initial state, which is fixed and taken in order,
and not on what the policy did. All twelve first-observation artifacts across
the three episodes - two camera frames, the end-effector state and the prompt -
are **byte-identical** between the CPU client and the CUDA control. This check
was designed after the failure above and is labelled as such.

## Predictions, against the outcome

| | prediction | outcome |
|---|---|---|
| 1 | `envclient-robomimic-cpu` reports CUDA: no | **confirmed** |
| 2 | `envclient-libero-cpu` under 3.5G | **confirmed** - 3.4G |
| 3 | both CPU cases import `robosuite` | **confirmed** |
| 4 | the behaviour check matches exactly | **falsified** - and so would any client have been |
| 5 | `envclient-libero` exceeds E1's 7.2G | **confirmed** - 7.8G |

## A defect this turned up

**No run of this system is reproducible.** The denoising noise is drawn
unseeded on every inference, and the server offers no way to fix it. Two runs
of the same configuration take different trajectories, which is why the three
runs above return 301, 300 and 296 steps.

This does not touch E11's accounting claim, which is that the server's record
of episodes and steps agrees with the clients' **within a run**. It does mean
that rerunning E11's Stage A or its evaluations will not return the same
success rates, and nothing in that experiment says otherwise. A `--policy.seed`
option would fix it and is not part of this experiment.

## What this does not support

- **Nothing about speed.** The CPU torch is not asked to compute here. If the
  environment side ever needs torch to do arithmetic, this install is the
  wrong one.
- **Not "torch is unnecessary".** It is still 711M of a framework the
  environment never calls, present because LIBERO and robomimic import it at
  module level. Removing it means patching those imports.
- **Not "no GPU anywhere".** Rendering went through EGL on the cluster's
  cards. A machine with no GPU at all is untested.
- **Not a success-rate claim.** Three episodes on one task, successful under
  both clients, says nothing about the rate on a suite.
- **One machine, one day.** Dependency resolution moves; the logs record what
  resolved on 2026-09-18.

## Budget

Four installs, 279 s in total. One evaluation run per client, about 4 minutes
each including model load. The exploratory copy that preceded the protocol took
13 minutes, most of it copying 35,813 files on Lustre.
