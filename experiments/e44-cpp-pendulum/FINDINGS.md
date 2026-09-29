# E44: a C++ program with no third-party libraries trains a policy on its own Pendulum, as well as the Python env client does

2026-09-28 · guangzhao (Linux workstation, CPU only) · protocol:
[`PROTOCOL.md`](PROTOCOL.md) (`70f8cfd`, after the pilot in
`results/pilot.txt`, before the registered run)

---

## The result

`pendulum_client.cpp`:
- 867 lines, built with g++ at -O2, linking only libstdc++, libm, libgcc and
  libc.
- Carries its own copy of gymnasium's Pendulum-v1.
- Trained gaussian-policy through plugrl-server with ppo at rl-baselines3-zoo's
  Pendulum settings.

The control is plugrl-env-client stepping gymnasium's Pendulum-v1 against the
same server. Mean return, iterations 1-2 against 16-25, over 25 iterations of
4,096 steps:

| seed | C++ client | Python env client |
| --- | --- | --- |
| 1 | -1,223 -> -251 (**+972**) | -1,191 -> -206 (**+984**) |
| 2 | -1,172 -> -233 (**+939**) | -1,188 -> -264 (**+924**) |
| 3 | -1,227 -> -275 (**+953**) | -1,259 -> -264 (**+996**) |

* **P1 holds.** All six seeds logged 25 iterations with no traceback, and
  every client exited 0.
* **V1 passes.** Each arm's clients are what the arm says. The six servers ran
  identical configurations with the registered settings.
* **P2 and P3 hold.** Both arms learn on 3 of 3 seeds against a bar of +500.
* **P4 holds.** The C++ seeds end at -251, -233 and -275, inside the control's
  -264 to -206 widened by 150. Only -275 falls outside the unwidened range,
  by 11.

The C++ client counts on its own and agrees with the server. Each seed ran
102,401 steps in 512 episodes, and its last ten episodes averaged -195, -161
and -210. rl-zoo reports -172 ± 104 for PPO at 100k steps. The C++ arm ran
iterations 1-25 in 146 s against the control's 194 s: no interpreter and no
wrapper sit between its environment and its socket.

---

## What makes the comparison fair

The C++ Pendulum is gymnasium's, step for step. `equivalence.py` ran 20
trials of 200 steps from random states under random torques, some outside
the clip:
- observations bit-identical
- rewards within 1.2e-10, below float32's resolution; the reward travels as
  float32
- the time limit on the same step every time

That took matching NumPy's precision rules. Under NumPy 2, `0.001 * u**2` and
`3.0 * u` are float32 products, because the torque arrives as float32; the
rest is float64. It also took NumPy's `%` in the angle's normalisation, not
C's `fmod`.

So the two arms differ only in who steps the environment and how the
exchange is packed. P4 is the check on the packing, where a misattributed
reward or an episode boundary one step off would pull the C++ arm below the
control. It found neither.

---

## What the pilot found

* **SPEC.md leaves out what the server's episode metrics read.**
  - The server's `rollout/reward`, `/length` and `/success` come only from
    `info["episode"]` (`r`, `l`, `s`, `mask`), which plugrl-env-client's
    `VectorEpisodeStatsWrapper` adds on an episode's last step. SPEC.md does
    not mention it.
  - The C++ client, written from the spec, sent `info {}`. The server trained
    it normally and logged a return of 0 for every iteration.
  - The client now sends `info["episode"]` as the wrapper does. Training was
    never affected, because it reads only the feedback's reward and flags.
    Anyone writing a client from the spec alone would hit this. The spec
    should say it.
* **CleanRL's default PPO settings barely learn Pendulum**, in both arms
  alike. In CleanRL's own 1M steps, seed 0 went from -1,194 to -1,149 (C++)
  and from -1,196 to -1,011 (Python).
  - This is PPO's defaults on this task, not either client. Pendulum wants a
    short horizon, and rl-zoo's settings use gamma 0.9.
  - The registered run used rl-zoo's published settings for Pendulum-v1
    (`hyperparams/ppo.yml`, last changed at eec15dc). Two things differ from
    them: one env collecting 4,096 steps an iteration where rl-zoo uses
    4 x 1,024, and no gSDE, which gaussian-policy has no counterpart for.
* **One run was lost to a port.** The first Python run of rl-zoo's settings
  started while the previous Python pilot was still finishing on the same
  port, and could not bind. It was rerun, and the registered run started only
  after checking that no port was held.

---

## What E44 does not show

* **A second environment, or a harder one.** Pendulum has a 3-number
  observation, one action, and 200-step episodes, with states only and no
  images.
* **An independent author.** The C++ client, like E2's, was written by the
  author of the spec it follows. It shows the spec plus a reading of the
  Python client suffice, and the pilot shows the spec alone does not: see the
  first item above.
* **More than one env per client, or action chunks.** The client runs one
  env and asks for an action every step, which is replan 1.
* **Any platform but Linux.** It uses POSIX sockets.
* **gSDE.** rl-zoo's -172 used it and this run did not. The comparison here
  is between the two clients under identical settings, not against rl-zoo.

Checkpoints and tensorboards stay on guangzhao. Kept here:
- the logs of both arms
- `verdicts.txt` and `summary.tsv` (`summarise.py`)
- `pilot.txt`
