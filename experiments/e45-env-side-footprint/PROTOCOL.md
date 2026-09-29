# E45 measurement protocol (pre-registered)

**Written 2026-09-29, after the harness pilot in `results/pilot.txt`, before
any registered measurement.**

This file must not be edited after the first registered data point. Anything
learned afterwards goes in `AMENDMENT.md`, dated.

---

## The question

PlugRL's claim is that the environment side of *training* can be thin: no
torch, no ML framework, even no Python (E44). Other systems also move an
environment away from its trainer. So a reader asks what separates PlugRL
from them.

**What does the environment side of each system have to install to exchange
training data with its trainer, and could it be written in another
language?**

---

## Declared in advance: what was already known

1. **The systems** are chosen by the related-work survey of 2026-09-28. For
   each, the environment side is the code that must run beside the
   environment.
   - **plugrl-env-client**: PlugRL's Python env client, with no environment
     extras, at 3ea0609.
   - **plugrl-minimal**: what plugrl-protocol's `examples/raw_client.py`
     imports, which is msgpack and websockets.
   - **plugrl-cpp**: E44's C++ client. It has no packages, because it is an
     86 KB binary linking libstdc++, libm, libgcc and libc. It is recorded
     from E44 and not installed here.
   - **rllib-rllink**: RLlib's external-env client on its new API stack,
     ray 2.58.0. Its reference client (`dummy_external_client.py`) imports
     `ray.rllib`, gymnasium and torch, runs the policy itself with torch, and
     builds that policy from an algorithm config the server sends pickled
     (`pickle.loads(msg_body["config"])`). The framing is msgpack.
   - **lerobot-hilserl**: LeRobot's HIL-SERL actor, lerobot 0.6.1.
     - `rl/actor.py` imports torch and runs the policy.
     - `transport/utils.py` moves weights and transitions as
       `torch.save`/`torch.load` bytes and other objects as pickle.
   - **dm-env-rpc**: DeepMind's environment protocol, 1.1.7, environment
     side. It is gRPC and protobuf. It has no trainer, and reward travels
     only by convention.
   - **openpi-client**: openpi's policy client at 215abfb. It is
     inference-only and carries no reward or done.
2. **E12**: a default Linux install of torch brings its CUDA wheels whether
   or not anything uses them. LIBERO's env client was 7.8G with sixteen
   nvidia wheels; with CPU torch it was 3.4G.
3. **The harness pilot** (plugrl-minimal only): 2 packages, 2.6 MB, no
   torch, the entry import works, 0.3 s. The pilot also fixed the package
   count, which had included uv's "Using Python" line.

---

## Design

`measure.sh NAME OUT` runs once per system, one system at a time, on
guangzhao (Linux, uv 0.12.6). The order is plugrl-minimal,
plugrl-env-client, dm-env-rpc, openpi-client, rllib-rllink, lerobot-hilserl.

Each system gets:
- a fresh venv with uv's managed Python (3.11; lerobot needs 3.12)
- a cold cache of its own

Each run records:
- the install's exit status and time
- the number of packages
- the venv's size in bytes; the interpreter is not counted, since every
  system shares a managed one
- how many `nvidia-*` packages it installed
- whether `import torch` works
- whether the system's entry module imports

The venv and its cache are deleted afterwards. The whole run is one script,
`run.sh`.

---

## Predictions, and what falsifies each

**P1 - torch.** plugrl-env-client, plugrl-minimal, dm-env-rpc and
openpi-client install no torch. rllib-rllink and lerobot-hilserl install
torch, and with it at least one `nvidia-*` package.

> Grounds: known items 1 and 2. Falsified by any system on the wrong side.

**P2 - size.** plugrl-env-client's venv is at least **10x** smaller than
each of rllib-rllink's and lerobot-hilserl's.

> Grounds: known item 2 puts torch's default install in the gigabytes, and
> plugrl-env-client's heaviest dependencies are gymnasium, pandas and
> imageio's ffmpeg binary. Falsified if either ratio is under 10.

**P3 - thin is not unique.** dm-env-rpc's and openpi-client's venvs are each
within **3x** of plugrl-env-client's, in either direction.

> Grounds: all three are protocol clients without a framework. This is the
> prediction against PlugRL's distinctiveness: if it holds, a thin
> environment side is not what sets PlugRL apart. What does is a thin side
> that trains (item 1: dm-env-rpc has no trainer, openpi-client carries no
> feedback). Falsified if either ratio is outside [1/3, 3].

**P4 - the measurements are of working installs.** Every system installs
with exit status 0 and its entry module imports.

> Falsified by any failure. A failure does not stop the others; it is
> reported with its log.

**Reported, not predicted:**
- package counts
- install times
- which of each system's environment-side messages could be produced
  without Python, from item 1's sources

---

## Declared deviations allowed in advance

1. One rerun of a system whose install fails for a reason outside the
   system, such as a network error, recorded in `AMENDMENT.md`.

---

## Reading order

P4, P1, P2, P3, then the reported figures.
