# E45: among systems that train through their channel, only PlugRL's environment side needs no torch - but a thin environment side alone is not unique

2026-09-29 · guangzhao (Linux, uv 0.12.6), a fresh venv and a cold cache per
system · protocol: [`PROTOCOL.md`](PROTOCOL.md) (`ae95e70`, after the harness
pilot in `results/pilot.txt`, before any registered measurement)

---

## The result

This is what each system's environment side installs in order to exchange
data with its trainer. The venv sizes leave out the interpreter, which every
system shares.

| system | trains through the channel | packages | size | torch | nvidia packages |
| --- | --- | --- | --- | --- | --- |
| **PlugRL**, C++ client (E44) | yes | 0 | 86 KB binary | no | 0 |
| **PlugRL**, minimal Python client | yes | 2 | 2.6 MB | no | 0 |
| **PlugRL**, `plugrl-env-client` | yes | 31 | 222 MB | no | 0 |
| RLlib external env (RLlink), ray 2.58.0 | yes | 64 | 6,105 MB | 2.14.0+cu130 | 15 |
| LeRobot HIL-SERL actor, 0.6.1 | yes | 144 | 6,142 MB | 2.11.0+cu130 | 15 |
| dm_env_rpc 1.1.7 | no trainer | 12 | 81 MB | no | 0 |
| openpi-client | no, inference only | 9 | 90 MB | no | 0 |

* **P4 is falsified.** openpi-client installed, but its entry module does not
  import on its own. `websocket_client_policy.py` imports `typing_extensions`,
  which openpi-client does not declare. Inside openpi's own workspace another
  package brings it, which is why it goes unnoticed. The other five installs
  work, and the measurement stands: the missing package is about 0.1 MB.
* **P1 holds.** Torch is installed only by the two systems whose environment
  side runs the policy, RLlib's and LeRobot's, and both install it with CUDA's
  15 wheels by default.
* **P2 holds.** Those two environment sides are 27.5x and 27.7x the size of
  plugrl-env-client.
* **P3 holds, and it counts against PlugRL.** dm_env_rpc and openpi-client
  are within 3x of plugrl-env-client, and smaller: 0.37x and 0.40x.

Every install took under 40 s on guangzhao's link, so install time
separates nothing here.

---

## What separates PlugRL, and what does not

**A thin environment side is not unique.** dm_env_rpc's is thinner than
plugrl-env-client, and so is openpi's. Neither trains through its channel.
dm_env_rpc has no trainer and carries reward only by convention. openpi
carries no reward or done.

**Thin and training together is.** Of the systems here that do train through
their channel, both others put torch and the policy on the environment side:
- **RLlib's reference client** builds the policy from an algorithm config
  the server sends pickled, and runs it with torch.
- **LeRobot's actor** runs the policy and moves weights and transitions as
  `torch.save` bytes.

That also settles the language question from their sources: neither can be
spoken without Python and torch, short of reimplementing their policy and
serialisation. PlugRL's C++ client (E44) trains a policy with nothing beyond
the C++ standard library.

**plugrl-env-client is heavier than it needs to be.** A reinstall measured
package by package shows where its 222 MB goes:
- **imageio's ffmpeg binary, 77 MB.** Only the episode recorder imports
  imageio.
- **pandas, 45 MB.** Nothing in the package imports it.
- **numpy, 59 MB.** Gymnasium needs it.
- **Pillow, 20 MB.** It arrives with imageio, and only LIBERO's image tools
  use it.

The protocol itself needs msgpack and websockets, 2.6 MB. Dropping pandas and
moving imageio to an extra would cut the default install by more than half,
to around dm_env_rpc's size.

---

## What E45 does not show

* **Anything at runtime.** This measures installs, not memory, latency, or
  whether each system trains well.
* **CPU-only torch.** RLlib's and LeRobot's sides were installed as their
  instructions do, with the default torch. E12 showed that choosing CPU torch
  roughly halves such an install, which would still leave them over ten times
  plugrl-env-client.
* **Other systems** the survey lists: SEED RL (archived), ML-Agents (tied to
  Unity), and RLlib's old PolicyClient.

The install logs, `uv pip list` for every system, the import errors, and
`footprint.tsv` are in `results/`.
