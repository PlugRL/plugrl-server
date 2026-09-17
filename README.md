# plugrl-server

[![CI](https://github.com/PlugRL/plugrl-server/actions/workflows/ci.yml/badge.svg)](https://github.com/PlugRL/plugrl-server/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The training side of PlugRL: it holds the policy and the learning algorithm,
batches inference across every connected env client over WebSocket, and learns
from the feedback those clients send back.

**A full-size pi0.5 has run end to end through it on LIBERO** - inference,
feedback and FPO training - with the server's record of episodes and steps
reconciling exactly with the clients'. As a control, the unmodified checkpoint
scored 99 of 100 on `libero_spatial` and 185 of 200 on `libero_10`, against
openpi's published 98.8 and 92.4.

**The reinforcement learning result is negative.** One FPO iteration took the
hardest task from 26 of 50 to 0 of 50, and the run is incomplete at one
iteration of ten: a second learn step does not fit beside the optimizer state
the first one allocates on a 24 GB card. The predictions were registered before
the run, and one of them is falsified.

| | Question | Answer |
|---|---|---|
| [E1](experiments/e1-dependency-conflict/) | Do a training stack and an environment stack really conflict? | **No** - the assumption this project was built on is disproved |
| [E6](experiments/e6-first-learning-curve/) | Does anything here actually learn? | **Yes** - FPO on HalfCheetah-v5, three seeds |
| [E7](experiments/e7-cross-machine/) | What does the boundary cost once packets leave the machine? | **+0.52 ms** on a 184 KiB observation |
| [E10](experiments/e10-vla-forward-cost/) | Is that cheap beside a VLA forward pass? | **Yes** - 1.3-3.6% of a step |
| [E11](experiments/e11-vla-rl-libero/) | Can a real VLA be trained through this boundary, and does it help? | **Ran end to end; did not help** |

[`experiments/`](experiments/) holds ten of these, nine of them run. Each
carries its data and a `FINDINGS.md` stating what the result does **not**
support. The documentation, including a quickstart that trains FPO on
HalfCheetah with no GPU and nothing to download, is at
[plugrl.github.io](https://plugrl.github.io).

## 🛠️ Installation

### For Users

Install the package with pip:

```bash
pip install -e .
```

### For Developers

We use `uv` to manage dependencies and development environments.

1. **Install uv** (if not already installed):

    ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

  Or use your package manager (e.g., `brew install uv` on macOS).

2. **Clone the repository:**

    ```bash
    git clone https://github.com/PlugRL/plugrl-server.git
    cd plugrl-server
    ```

  Nothing in the default install needs a git submodule. `pi0-policy` does -
  see [Training PI0 (OpenPI) with FPO](#training-pi0-openpi-with-fpo)
  for the extra steps a plain clone does not give you.

3. **Install dependencies with uv:**

    ```bash
  uv sync
    ```

  This will create a virtual environment at `.venv/` and install all dependencies specified in `pyproject.toml`.

4. **Run commands via uv:**

    ```bash
  uv run python -m plugrl_server.cli --help
    ```

### 🚀 Usage

`plugrl-server` provides command-line utilities to run RL training with different algorithms and policies. The codebase supports both local training with WebSocket-based agent communication and distributed training with Ray.

#### Available Components

**Algorithms:**
- `fpo` - Flow Policy Optimization. **The one that learns with no extras installed.**
- `dummy` - Dummy algorithm for testing and debugging. Its `learn` is a
  `sleep`; it moves no weights.
- `dppo` - DPPO (Diffusion Policy Policy Optimization). Requires
  `plugrl-server[dppo]`.
- `dppo-dist` - Distributed DPPO (experimental)
- `eval` - Evaluation only, no learning

**Policies:**
- `fpo-policy` - Flow policy. Defaults to `obs_dim=17`, `action_dim=6`
- `dummy-policy` - Outputs random actions (for testing)
- `dppo-policy` - DPPO policy (requires `plugrl-server[dppo]` and a checkpoint)
- `pi0-policy` - PI0 policy (OpenPI). Needs more than a checkpoint. The
  `openpi` package is not on PyPI and no extra installs it: it is the
  `third_party/openpi` git submodule, which `git clone` does not fetch. See
  [Training PI0 (OpenPI) with FPO](#training-pi0-openpi-with-fpo).

`python -m plugrl_server.cli --help` lists the **policies** that are actually
available in your install - with no extras it prints `{dummy-policy,fpo-policy}`,
and `dppo-policy` and `pi0-policy` are simply absent.

**Algorithms do not behave the same way, and `dppo` is the trap.** With no
`dppo` package installed, `... fpo-policy default --help` still offers
`{fpo,dummy,eval,dppo}`: the DPPO *config* module imports cleanly and
registers its configs, while the module carrying the algorithm class does not.
Selecting `dppo` parses, starts up, prints the config, and then dies with
`KeyError: 'Algorithm dppo is not registered.'`. The real signal is the
warning on the very first line of every command,
`Could not import DPPO algorithm module for reason: No module named 'dppo'`.
(`dppo-dist` is genuinely absent in that state, so the menu is inconsistent
with itself.)

#### Quick Start: a run that actually learns

FPO on HalfCheetah-v5, CPU only, no extras beyond `plugrl-env-client[mujoco]`.
HalfCheetah-v5 has a 17-dimensional observation and a 6-dimensional action,
which are exactly `fpo-policy`'s defaults, so nothing needs configuring.

```bash
# Terminal 1: the training server
python -m plugrl_server.cli fpo-policy default fpo default \
    --port 8000 --policy.device cpu \
    --algo.global-steps 500000 --algo.buffer-size 4096

# Terminal 2: the environment (from plugrl-env-client)
python -m plugrl_env_client.cli mujoco-v1 \
    --server-port 8000 --num-envs 1 --num-episodes 600 \
    --runner.replan-steps 1 --runner.seed 0
```

Episode return starts near -300. Across three seeds it is still dipping back
into the -300s at step 20k, the mean crosses zero at about 60k, and by 500k
steps it reaches 1928 +/- 224 - roughly a hundred minutes on the CPU-only
laptop that measured it. The first few minutes are noise; judge it over tens
of thousands of steps. `experiments/e6-first-learning-curve/` has the
three-seed curve, the logs and the findings.

**`--algo.buffer-size` is not optional here, and the default will surprise
you.** FPO learns when its rollout buffer fills *or* when the run reaches its
last step. At the default `buffer_size=983040`, any run shorter than about a
million steps therefore learns exactly once, at the very end - producing a
single point rather than a curve. 4096 gives one update per 4096 environment
steps.

#### Quick Start: Testing with Dummy Components

To test the agent-server connection with dummy algorithm and policy:

```bash
# Terminal 1: Start the server
python -m plugrl_server.cli dummy-policy default dummy default

# Terminal 2: Start the environment client (from plugrl-env-client)
# The server will listen on localhost:8000 by default
```

**Expected output from server:**
```
12:34:56|INFO|plugrl_server version: 0.1.0
12:34:56|INFO|Algorithm: dummy, Config: DummyAlgoConfig(...)
12:34:56|INFO|Policy: dummy-policy, Config: DummyPolicyConfig(...)
12:34:56|INFO|Checkpoint Manager created:
<...CheckpointManager object...> at ./checkpoints/dummy/dummy-policy/...
12:34:57|INFO|Policy created...
12:34:57|INFO|Algorithm created:
<...DummyAlgorithm object...>
12:34:57|INFO|Agent Server is listening on 0.0.0.0:8000
```

#### Training DPPO with DPPO Policy

```bash
# Single GPU training with WebSocket server
python -m plugrl_server.cli dppo-policy default dppo hopper \
  --exp_name my_dppo_exp \
  --track.enabled

# With custom learning rates and batch size
python -m plugrl_server.cli dppo-policy default dppo hopper \
  --algo.actor_lr 5e-5 \
  --algo.batch_size 256 \
  --exp_name my_dppo_exp_custom
```

**Configuration options:**
- `--algo.actor_lr` - Actor learning rate (default varies by environment)
- `--algo.critic_lr` - Critic learning rate
- `--algo.batch_size` - Batch size for training
- `--algo.train_itrs` - Number of training iterations (default 200). This is
  the one that ends the run: DPPO stops when `curr_train_itrs` reaches it, and
  `global_steps` is derived from it as `train_itrs * buffer_size`.
- `--checkpoint_base_dir` - Directory to save checkpoints (default: `./checkpoints`)
- `--exp_name` - Experiment name (auto-generated if not provided)
- `--track.enabled` - Enable experiment tracking with SwanLab or W&B. A bare
  flag, not a value-taking option: `--track.enabled true` is a tyro parse
  error (`Unrecognized arguments: true`). The off switch is
  `--track.no-enabled`.
- `--track.tracker` - Tracker type: `swanlab` (default) or `wandb`

**Do not use `--algo.n_train_itr`.** The `hopper` and `libero` variants carry
an `n_train_itr` field that nothing ever reads, so the flag parses and is
silently ignored. (`n_critic_warmup_itr` is dead in the same way; the live
field is `n_critic_warmup_itrs`.) Earlier versions of this README documented
`--algo.n_train_itr` as "number of training iterations", which was wrong.

#### Training PI0 (OpenPI) with FPO

**Prerequisites, which no earlier step in this README provides.** `pi0-policy`
imports `openpi` (`from openpi import transforms`,
`openpi.training.checkpoints`, `openpi.models_pytorch.pi0_pytorch`, ...).
`openpi` is not on PyPI, and `uv sync --extra openpi` does not supply it - that
extra only adds `huggingface-hub`, `safetensors` and `lerobot`. The package
comes from the `third_party/openpi` git submodule:

```bash
git submodule update --init third_party/openpi
uv pip install -e third_party/openpi/packages/openpi-client
uv pip install -e third_party/openpi
uv sync --extra openpi
```

`src/plugrl_server/policy/openpi/README.md` carries the same sequence plus a
`transformers` file-replacement step
(`cp -r third_party/openpi/src/openpi/models_pytorch/transformers_replace/* ...`)
that openpi's PyTorch path requires. Until all of this is done, `pi0-policy`
is not in the `python -m plugrl_server.cli --help` menu at all - the policy
package catches the import error and registers nothing, so the failure is
silent.

**These steps have since been run.** E11 installed openpi on a single RTX
3090 and ran `pi0-policy` end to end, both evaluation and FPO training, against
a full-size `pi05_libero` checkpoint. The resolved environment of both
processes is recorded in
`experiments/e11-vla-rl-libero/results/environment-server.txt` and
`environment-client.txt`, and the pins that mattered, with the reasons, are in
`experiments/e11-vla-rl-libero/NOTES.md`. What E6's findings file says - that
the openpi path had never been executed - was true when it was written.

The commands below are the ones E11 ran, with the checkpoint path left for you
to fill in.

```bash
# Evaluation: no learning, one server for as many env clients as you start
python -m plugrl_server.cli pi0-policy default eval default \
  --policy.name pi05_libero \
  --policy.checkpoint-path /path/to/pi0/checkpoint \
  --policy.device cuda

# FPO fine-tuning, as E11 ran it
python -m plugrl_server.cli pi0-policy default fpo default \
  --policy.name pi05_libero \
  --policy.checkpoint-path /path/to/pi0/checkpoint \
  --policy.device cuda \
  --algo.learning-rate 1e-5 --algo.batch-size 8 \
  --algo.n-samples-per-action 4 --algo.buffer-size 4096 \
  --algo.global-steps 40960
```

`pi0-policy` is a flow policy, so it pairs with `fpo` or with `eval`. It does
not pair with `dppo`, which expects a diffusion policy - `hopper` is one of
that algorithm's presets, and this README used to pass both.

The batch size of 8 is not the protocol's starting value of 32. At 32 the first
learn step ran out of memory on a 24 GB card, which
`experiments/e11-vla-rl-libero/AMENDMENT.md` records. Even at 8, the *second*
learn step does not fit beside the optimizer state the first one allocates;
`experiments/e11-vla-rl-libero/FINDINGS.md` has the measurements.

**PI0 policy options:**
- `--policy.checkpoint-path` - Path to PI0 checkpoint directory (required)
- `--policy.name` - openpi training config, for example `pi05_libero`
- `--policy.denoising-steps` - Number of denoising steps (default: 5)
- `--policy.train-expert-only` - Freeze VLM and train only expert (default: true)

#### Distributed Training with Ray (DPPO)

For multi-GPU distributed training, use the Ray launcher.

**Only `dppo-dist` is launchable under Ray, not `dppo`.** `cli_ray` asserts
`isinstance(algo, DDPAlgorithm)` right after building the algorithm, and
`DPPOAlgorithm` does not inherit `DDPAlgorithm` - only
`DPPOAlgoDistributed` (`dppo-dist`) does. Passing `dppo` here fails with
`AssertionError: Algorithm dppo is not a DDPAlgorithm.` before Ray does any
work. `fpo` fails the same assertion, so FPO cannot run on this path at all
(`experiments/e6-first-learning-curve/FINDINGS.md`). Earlier versions of this
README used `dppo` in all three commands below.

`dppo-dist` needs `plugrl-server[dppo]` like `dppo` does; without it the
variant does not register and the command cannot be typed.

```bash
# Multi-GPU distributed training
python -m plugrl_server.cli_ray dppo-policy default dppo-dist hopper \
  --num_ddp_gpus 4 \
  --exp_name dppo_dist_4gpu

# All available GPUs
python -m plugrl_server.cli_ray dppo-policy default dppo-dist hopper \
  --exp_name dppo_dist_all_gpus

# With specific inference GPU
python -m plugrl_server.cli_ray dppo-policy default dppo-dist hopper \
  --infer_gpu 0 \
  --num_ddp_gpus 4
```

**Ray-specific options:**
- `--infer_gpu` - GPU index for inference (default: first GPU)
- `--num_ddp_gpus` - Number of GPUs for DDP training (default: all available)
- `--master_addr` - Master address for DDP (default: auto-detected)
- `--master_port` - Master port for DDP (default: auto-assigned)

#### Resuming Training

To resume a previous training run:

```bash
# This will restore from the latest checkpoint in the checkpoint directory
python -m plugrl_server.cli dppo-policy default dppo hopper \
  --exp_name my_dppo_exp \
  --resume
```

`--resume` is a bare flag. `--resume true` is a tyro parse error
(`Unrecognized arguments: true`); the off switch is `--no-resume`, which is
also the default.

#### Viewing All Available Configurations

To see all registered configurations:

```bash
# This will show the nested menu structure
python -m plugrl_server.cli --help
```

Output will show available combinations of algorithms and policies with their variants.

### Seven corrections to earlier versions of this README

Checked 2026-09-11 against this repository's `.venv` and `src/`. Each
corrected command was run here, with one stated exception: the openpi install
steps in item 6 were not, and are labelled unverified where they appear.

1. **`--track.enabled true` and `--resume true` never worked.** tyro renders
   booleans as paired flags, so the literal `true` is a separate, unrecognized
   argument. All three commands that used this form failed at parse time with
   `Unrecognized arguments: true`. They now read `--track.enabled` and
   `--resume`.
2. **The Ray examples named an algorithm Ray rejects.** All three used `dppo`,
   which trips `assert isinstance(algo, DDPAlgorithm)` in `cli_ray.py` before
   any Ray work happens. They now use `dppo-dist`, the only DPPO variant that
   satisfies it.
3. **`--algo.n_train_itr` does nothing.** It is a dataclass field on the
   `hopper` and `libero` variants that no code reads. The live control is
   `--algo.train_itrs`.
4. **The expected `dummy` output showed a line the server does not print.**
   There is no `WebSocket server listening on ...` anywhere in `src/`; the
   real line is `Agent Server is listening on 0.0.0.0:8000`. The block now
   matches a captured run.
5. **"the optional ones do not appear without their extra" was false for
   `dppo`.** It is true for the optional policies and false for that one
   algorithm, which appears in the menu and then fails at startup. The text
   now says which is which.
6. **`pi0-policy` was described as needing only a checkpoint.** It also needs
   the `third_party/openpi` git submodule, which no install step in this
   README mentioned. The submodule steps are now documented, and marked
   unverified because they are.
7. **The learning-curve timing said "CPU-only desktop".** The machine that
   measured it was a laptop (Intel Iris Xe, no NVIDIA GPU); see
   `experiments/e6-first-learning-curve/FINDINGS.md`.
