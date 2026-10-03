# plugrl-server

[![CI](https://github.com/PlugRL/plugrl-server/actions/workflows/ci.yml/badge.svg)](https://github.com/PlugRL/plugrl-server/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

The training side of PlugRL: it holds the policy and the learning algorithm,
batches inference across every connected env client over WebSocket, and learns
from the feedback those clients send back.

<img src="https://plugrl.github.io/media/libero-demo.gif" width="360"
     alt="A Franka arm in LIBERO reaching for and picking up an object, seen from the policy's own camera">

*The **unmodified** pi0.5, driven through this server on `libero_spatial` task
0 - one episode of the three that ran, all three successful. These are the
frames the env client sends as observations, at the policy's native 224x224,
not an outside camera.*

The env clients can run somewhere else: in another process, on another
machine, on a machine with no GPU, or in a program that is not Python. What
has been measured:

| | Question | Answer |
|---|---|---|
| [coverage](#what-runs-on-it) | Which policies and algorithms learn through the split? | **Every one tried** - sixteen combinations on four tasks, all learning, below |
| [E44](experiments/e44-cpp-pendulum/) | Does an env client have to be this codebase, or Python? | **No** - a C++ program with no third-party libraries trains a policy on its own Pendulum as well as the Python env client does ([E2](experiments/e2-cross-language/) first spoke the protocol from C++) |
| [E45](experiments/e45-env-side-footprint/) | And the other systems that train through their channel? | **Torch on the env side** - RLlib's external-env client and LeRobot's HIL-SERL actor install 6.1 GB each, 27x plugrl-env-client. dm_env_rpc and openpi-client are thin too, but do not train |
| [E12](experiments/e12-cuda-free-rollout/) | Does a rollout machine need CUDA, or a GPU? | **No** - LIBERO's env client goes from 7.8G to 3.4G with no nvidia wheels, and renders on the CPU at 1.91x the wall clock ([E13](experiments/e13-gpu-free-rendering/)) |
| [E43](experiments/e43-cross-machine-training/) | Does training still work with the env clients on another physical machine? | **Yes** - the quickstart pair learns with its env clients on a Windows laptop over campus Wi-Fi |
| [E46](experiments/e46-reuse-feedback-obs/) | What does crossing cost? | About **3 ms plus the observation's bytes over the link** per exchange: 11.6 ms for 184 KiB at 23.7 MB/s. Each observation crosses once with `reuse-feedback-obs`, which the server offers and the env client uses by default; [E43](experiments/e43-cross-machine-training/) measured twice the bytes before it, and the weights are byte-identical either way |
| [E10](experiments/e10-vla-forward-cost/) | Is that cheap beside a VLA forward pass? | **On a fast link.** On that link a 184 KiB observation is 12% of pi0.5's 100 ms forward, not E10's 1.3-3.6% |
| [E49](experiments/e49-multi-trainer/) | Do other trainers train through it? | **Yes** - RLinf, Stable-Baselines3 and CleanRL train PlugRL env clients through `plugrl-bridges` (not public yet), 3 of 3 seeds in every pairing; RLinf also with the env clients on a laptop with no torch, Ray or RLinf ([E48](experiments/e48-rlinf-bridge/)) |
| [E50](experiments/e50-boundary-transparency/) | Does training see the boundary? | **No** - SB3 with its 16 environments behind the protocol ends on byte-identical weights to SB3 with them in its own process, also on Atari frames ([E53](experiments/e53-image-observations/)) and with 25 ms each way between the two ([E52](experiments/e52-latency-sweep/)). It costs 0.4-0.6 ms per vector step on one machine |
| [E11](experiments/e11-vla-rl-libero/) | Does a full-size VLA go through it? | **Yes** - pi0.5 on LIBERO scores what openpi publishes, and the server's episode and step counts match the clients' exactly |

Fine-tuning pi0.5 with reinforcement learning through it has not made the
policy better yet; that record is on
[its own page](https://plugrl.github.io/vla/).
[`experiments/`](experiments/) holds over fifty experiments. Each that has
run carries its data and a `FINDINGS.md` stating what the result does
**not** support. The documentation, including a quickstart that trains FPO
on HalfCheetah with no GPU and nothing to download, is at
[plugrl.github.io](https://plugrl.github.io).

## What runs on it

<a href="https://plugrl.github.io/#what-runs-on-it"><img src="https://plugrl.github.io/media/coverage-grid.jpg" width="100%"
   alt="Sixteen cells, four policy-algorithm pairs on HalfCheetah, Hopper, Walker2d and robomimic square, each with a frame from its trained policy, a training curve and a status. Every pair learns every task. On square each starts from a pretrained policy, and fpo-policy with FPO passes the bar on two of three seeds."></a>

Every combination of the two MLP policies and the two algorithms on four
tasks, and the baseline they are measured against: a Gaussian MLP with PPO.
That row is CleanRL's policy and PPO on the MuJoCo tasks and DPPO's on
square, each run as its authors run it. [On the project page](https://plugrl.github.io/#what-runs-on-it) each
cell plays its clip and shows the two commands that trained it, and pi0.5 on
LIBERO has [a page of its own](https://plugrl.github.io/vla/). Moving between
cells means changing a few words. On
Hopper, the four servers were:

```bash
plugrl-run-server fpo-policy default fpo default --policy.obs-dim 11 --policy.action-dim 3 --algo.buffer-size 4096 --algo.global-steps 409600
plugrl-run-server fpo-policy default dppo hopper --policy.obs-dim 11 --policy.action-dim 3 --algo.buffer-size 4096 --algo.train-itrs 100
plugrl-run-server dppo-policy hopper dppo hopper --algo.buffer-size 1024 --algo.batch-size 512 --algo.train-itrs 100
plugrl-run-server gaussian-policy default ppo default --policy.obs-dim 11 --policy.action-dim 3
```

The env client is the same for all four, except that DPPO's own policy acts
in chunks of four (`--runner.replan-steps 4`). None of these servers has
MuJoCo, robosuite or gymnasium installed. The env clients have them, in three
separate environments: MuJoCo 3 for these tasks, and MuJoCo 2.3.7 with
robosuite 1.4.1 for robomimic and LIBERO.

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

    This creates a virtual environment at `.venv/` and installs everything
    `pyproject.toml` asks for.

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
- `dppo` - DPPO (Diffusion Policy Policy Optimization). No extras needed,
  and it runs against `fpo-policy`.
- `dppo-dist` - Distributed DPPO (experimental)
- `ppo` - PPO as CleanRL's `ppo_continuous_action.py` runs it, every
  default included. Drives `gaussian-policy`; no extras needed.
- `eval` - Evaluation only, no learning

**Policies:**
- `fpo-policy` - Flow policy. Defaults to `obs_dim=17`, `action_dim=6`
- `gaussian-policy` - CleanRL's Gaussian MLP: tanh layers of 64, a log std
  that does not depend on the observation. Defaults to `obs_dim=17`,
  `action_dim=6`
- `dummy-policy` - Outputs random actions (for testing)
- `dppo-policy` - DPPO policy (requires `plugrl-server[dppo]` and a checkpoint)
- `pi0-policy` - PI0 policy (OpenPI). Needs more than a checkpoint. The
  `openpi` package is not on PyPI and no extra installs it: it is the
  `third_party/openpi` git submodule, which `git clone` does not fetch. See
  [Training PI0 (OpenPI) with FPO](#training-pi0-openpi-with-fpo).

`python -m plugrl_server.cli --help` lists the **policies** that are actually
available in your install - with no extras it prints `{dummy-policy,fpo-policy}`,
and `dppo-policy` and `pi0-policy` are simply absent.

**`dppo` used to be a trap and no longer is.** With no `dppo` package
installed, `... fpo-policy default --help` offered `{fpo,dummy,eval,dppo}` and
selecting `dppo` parsed, started up, printed its config and died with
`KeyError: 'Algorithm dppo is not registered.'` - the config module imported
cleanly and registered its configs while the module carrying the algorithm
class was never imported at all. Every command also began with
`Could not import DPPO algorithm module for reason: No module named 'dppo'`.

Both causes are gone. The algorithm reached into the `dppo` package for two
utilities - a running mean/variance and a learning-rate schedule - and neither
is DPPO-specific; both are MIT and are carried in
`algorithm/dppo/third_party/`, keeping their upstream file names and licence
headers. `algorithm/__init__.py` now imports the classes and not only their
configs. `dppo` and `dppo-dist` are registered on a plain install, and
`fpo-policy default dppo cheetah` runs on HalfCheetah-v5 with nothing beyond
`plugrl-env-client[mujoco]`.

`dppo-policy` is the part that still needs `plugrl-server[dppo]`: it wraps
DPPO's own `DiffusionModel` and builds it through DPPO's hydra configs, which
is a dependency on that project rather than on two utility classes.

Seven config fields went with it, for the same reason the menu entry did:
they were on the CLI and no code read them. `n_train_itr` and
`n_critic_warmup_itr` were DPPO's own yaml spellings of `train_itrs` and
`n_critic_warmup_itrs`, declared in six places and never reconciled, so
`--algo.n-train-itr 500` parsed, printed itself in the config banner and
changed nothing. `critic_batch_size` had a default and a `__post_init__` and
no reader; upstream DPPO does give the critic its own minibatch size, and
wiring that up is a change with its own validation rather than a rename.
`tests/test_dppo_without_the_extra.py` asserts that every field on every
`dppo` and `dppo-dist` config variant is read somewhere, so the next one is
caught rather than shipped.

A flow policy has no tractable density, which is the reason FPO exists, so it
is worth saying why DPPO can drive one at all: `sampling_noise_level` turns
each denoising step into a Gaussian transition, and DPPO's per-step
log-probability is that Gaussian's. At `sampling_noise_level=None` the
log-probability is exactly zero and DPPO would have nothing to form a ratio
from. The shipped `dppo` configs all set it.

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

#### The baseline: a Gaussian policy with PPO

The pair every other one is measured against, written to CleanRL's
`ppo_continuous_action.py`: a rollout of 2048 steps, ten epochs of 32
minibatches, clip 0.2, a learning rate of 3e-4 annealed to zero over 488
iterations (a million steps), observations and rewards normalised. CleanRL
does the normalising and the action clipping in gymnasium wrappers; PlugRL's
client does not wrap, so the policy and the algorithm do it on the server.

```bash
# Terminal 1: Hopper-v5 has an 11-dimensional observation and 3 actions
python -m plugrl_server.cli gaussian-policy default ppo default \
    --port 8000 --policy.device cpu \
    --policy.obs-dim 11 --policy.action-dim 3

# Terminal 2
python -m plugrl_env_client.cli mujoco-v1 \
    --server-port 8000 --num-envs 1 --num-episodes 100000 \
    --env.name Hopper-v5 --runner.replan-steps 1 --runner.seed 0
```

#### Quick Start: Testing with Dummy Components

To test the agent-server connection with dummy algorithm and policy:

```bash
# Terminal 1: Start the server
python -m plugrl_server.cli dummy-policy default dummy default

# Terminal 2: from plugrl-env-client, three short episodes against it.
# The server listens on 0.0.0.0:8000; a client connects to 127.0.0.1.
plugrl-run-env-client dummy-v1 --server-host 127.0.0.1 --num-episodes 3
```

The dummy policy's default action, continuous and 7-dimensional, is what
`dummy-v1` expects, so the pair needs no flags. The client exits 0 after its
three episodes.

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
