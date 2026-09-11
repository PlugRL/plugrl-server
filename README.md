# 🚀 plugrl-server

[![CI](https://github.com/PlugRL/plugrl-server/actions/workflows/ci.yml/badge.svg)](https://github.com/PlugRL/plugrl-server/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**plugrl-server** runs the policy and the learning algorithm, batching inference across every connected env client over WebSocket.

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
- `pi0-policy` - PI0 policy (OpenPI) (requires a checkpoint)

`python -m plugrl_server.cli --help` lists what is actually available in your
install; the optional ones do not appear without their extra.

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

Episode return climbs out of the -300s within about 40k steps, which takes a
few minutes on a laptop. `experiments/e6-first-learning-curve/` has the full
three-seed run and its findings.

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
12:34:56|INFO|plugrl_server version: X.X.X
12:34:56|INFO|Algorithm: dummy, Config: DummyAlgoConfig(...)
12:34:56|INFO|Policy: dummy-policy, Config: DummyPolicyConfig(...)
12:34:56|INFO|WebSocket server listening on 0.0.0.0:8000
```

#### Training DPPO with DPPO Policy

```bash
# Single GPU training with WebSocket server
python -m plugrl_server.cli dppo-policy default dppo hopper \
  --exp_name my_dppo_exp \
  --track.enabled true

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
- `--algo.n_train_itr` - Number of training iterations
- `--checkpoint_base_dir` - Directory to save checkpoints (default: `./checkpoints`)
- `--exp_name` - Experiment name (auto-generated if not provided)
- `--track.enabled` - Enable experiment tracking with SwanLab or W&B
- `--track.tracker` - Tracker type: `swanlab` (default) or `wandb`

#### Training DPPO with PI0 (OpenPI) Policy

```bash
# Single GPU training with PI0 policy
python -m plugrl_server.cli pi0-policy default dppo hopper \
  --policy.checkpoint_path /path/to/pi0/checkpoint \
  --exp_name pi0_dppo_exp \
  --track.enabled true

# With custom denoising steps
python -m plugrl_server.cli pi0-policy default dppo hopper \
  --policy.checkpoint_path /path/to/pi0/checkpoint \
  --policy.denoising_steps 10 \
  --exp_name pi0_dppo_exp_custom
```

**PI0 policy options:**
- `--policy.checkpoint_path` - Path to PI0 checkpoint directory (required)
- `--policy.denoising_steps` - Number of denoising steps (default: 5)
- `--policy.train_expert_only` - Freeze VLM and train only expert (default: true)

#### Distributed Training with Ray (DPPO)

For multi-GPU distributed training, use the Ray launcher:

```bash
# Multi-GPU distributed training
python -m plugrl_server.cli_ray dppo-policy default dppo hopper \
  --num_ddp_gpus 4 \
  --exp_name dppo_dist_4gpu

# All available GPUs
python -m plugrl_server.cli_ray dppo-policy default dppo hopper \
  --exp_name dppo_dist_all_gpus

# With specific inference GPU
python -m plugrl_server.cli_ray dppo-policy default dppo hopper \
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
  --resume true
```

#### Viewing All Available Configurations

To see all registered configurations:

```bash
# This will show the nested menu structure
python -m plugrl_server.cli --help
```

Output will show available combinations of algorithms and policies with their variants.
