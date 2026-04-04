# 🚀 plugrl-server

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
    git clone git@github.com:CTP314/plugrl-server.git
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

### Local OpenPI Source Install

For RoboCasa + OpenPI, the recommended workflow is to editable-install your existing local `/mnt/openpi-base` checkout into the current `plugrl-server` environment instead of manually editing files inside `.venv/`.

From the `plugrl-server` repo:

```bash
uv run python scripts/install_openpi_local.py --openpi-base-dir /mnt/openpi-base
```

What this does:

- editable-installs `/mnt/openpi-base/packages/openpi-client`
- editable-installs `/mnt/openpi-base`
- copies `openpi-base/src/openpi/models_pytorch/transformers_replace/*` into the current environment's `transformers/` package

The installer prefers `uv pip --python <current-venv-python>` so it also works in uv-managed environments that do not ship `pip` inside `.venv/`.

If you already know the current environment has all OpenPI dependencies and you only want to relink the local source tree, you can use:

```bash
uv run python scripts/install_openpi_local.py --openpi-base-dir /mnt/openpi-base --skip-deps
```

After installation, `plugrl-server` can load OpenPI directly from the local source checkout while keeping all policy-side RoboCasa integration logic in this repo.

If you plan to run `DPPO + OpenPI`, also make sure the optional DPPO dependency is present in the same environment:

```bash
uv sync --extra dppo
```

### 🚀 Usage

`plugrl-server` provides command-line utilities to run RL training with different algorithms and policies. The codebase supports both local training with WebSocket-based agent communication and distributed training with Ray.

#### Available Components

**Algorithms:**
- `dummy` - Dummy algorithm for testing and debugging
- `dppo` - DPPO (Diffusion-based Policy Optimization) algorithm
- `dppo-dist` - Distributed DPPO (experimental)

**Policies:**
- `dummy-policy` - Dummy policy that outputs random actions (for testing)
- `dppo-policy` - DPPO policy (requires `plugrl-server[dppo]` and checkpoint)
- `pi0-policy` - PI0 policy (OpenPI) (requires checkpoint)

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

#### Training DPPO with OpenPI RoboCasa Policy

The RoboCasa path extends the existing `pi0-policy` without changing any algorithm implementation. The policy side now:

- repacks `plugrl-env-client` RoboCasa observations into OpenPI RoboCasa inputs
- loads RoboCasa norm stats from an explicit path, shared config path, or `dataset_dir/{meta/,}` fallback
- can export first-batch policy inputs, including tokenized prompt arrays and decoded prompt text
- logs the checkpoint path, norm-stats source, and transform pipeline

Example:

```bash
python -m plugrl_server.cli pi0-policy robocasa dppo default \
  --host 127.0.0.1 \
  --port 8000 \
  --policy.name pi05_robocasa_test \
  --policy.checkpoint_path /path/to/openpi/checkpoint \
  --policy.dataset_dir /path/to/robocasa/lerobot_dataset \
  --policy.export_debug_artifacts \
  --policy.debug_artifact_dir ./debug/openpi_robocasa_inputs \
  --exp_name robocasa_dppo_openpi
```

If your RoboCasa config uses a shared norm-stats path in OpenPI config, `--policy.dataset_dir` can still be omitted. For single-dataset RoboCasa configs, `--policy.dataset_dir` is the expected option.

The policy debug directory will contain artifacts such as:

- `tokenized_prompt.npy`
- `tokenized_prompt_mask.npy`
- `tokenized_prompt_decoded.txt`
- raw input images and transformed images as `.png`
- raw/transformed state arrays as `.npy`
- `summary.json`

#### RoboCasa Env Client Example

From `/mnt/plugrl/plugrl-env-client`:

```bash
uv run plugrl-run-env-client robocasa-v1 \
  --num-envs 1 \
  --num-procs 1 \
  --num-episodes 1 \
  --server-host 127.0.0.1 \
  --server-port 8000 \
  --env.task_name SearingMeat \
  --env.split target \
  --env.action_encoding passthrough \
  --recorder.episode_freq 1 \
  --recorder.record_video \
  --recorder.record_full_rollout \
  --recorder.record_debug_packets
```

This records full-rollout mp4 files under the env-client run directory, which is the primary artifact for visually checking real environment interaction.

#### Current Scope Note

This integration round only supports `DPPO + RoboCasa + OpenPI` inside `plugrl-server`.

`FPO` is intentionally not wired up here because the current `FPO` implementation in `plugrl-server` still assumes array-like exported observations in its rollout train-state path, while OpenPI RoboCasa exports mapping-structured observations. Per the approved scope, no algorithm files were modified in this round.

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

## Testing

### Unit Tests

The RoboCasa/OpenPI policy-side helper coverage lives in:

- [tests/test_openpi_robocasa_helpers.py](/mnt/plugrl/plugrl-server/tests/test_openpi_robocasa_helpers.py)

It checks:

- RoboCasa observation repacking
- norm-stats source selection
- policy debug artifact export, including decoded tokenized prompt output

### Manual End-to-End Test

The real RoboCasa/OpenPI/DPPO roundtrip test lives in:

- [tests/test_openpi_robocasa_manual.py](/mnt/plugrl/plugrl-server/tests/test_openpi_robocasa_manual.py)

Before running it, export:

```bash
export OPENPI_ROBOCASA_CONFIG_NAME=...
export OPENPI_ROBOCASA_CHECKPOINT_DIR=...
export OPENPI_ROBOCASA_DATASET_DIR=...
```

And make sure the server-side Python environment can import `dppo`. If it cannot, the manual pytest case will skip with an explicit message instead of failing mid-run.

Optional overrides:

```bash
export PLUGRL_SERVER_OPENPI_PYTHON=/mnt/openpi-base/.venv/bin/python
export PLUGRL_ROBOCASA_ENV_CLIENT_PYTHON=/mnt/plugrl/plugrl-env-client/.venv/bin/python
export PLUGRL_ROBOCASA_MANUAL_MAX_EPISODE_STEPS=40
```

Run:

```bash
pytest -m manual tests/test_openpi_robocasa_manual.py::test_manual_openpi_robocasa_dppo_roundtrip -s
```

Artifacts to inspect after the manual test:

- env rollout videos: `/mnt/plugrl/plugrl-env-client/runs/<exp_name>/rollout/proc_000/videos/full/images/*.mp4`
- policy input dump: pytest temp dir under `policy_debug/first_observation/`
- server logs: pytest temp dir `server.log`
