# 🚀 VLARL-Launcher

## 🛠️ Installation

### For Users

Install the package with pip:

```bash
pip install -e .
```

### For Developers

We use Poetry to manage dependencies and development environments.

1. **Install Poetry** (if not already installed):

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Or use your package manager (e.g., `brew install poetry` on macOS).

2. **Clone the repository:**

    ```bash
    git clone git@github.com:CTP314/vlarl-launcher.git
    cd vlarl-launcher
    ```

3. **Install dependencies with Poetry:**

    ```bash
    poetry install
    ```

    This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

4. **Activate the Poetry environment:**

    ```bash
    poetry shell
    ```

    Or run commands directly with `poetry run`:

    ```bash
    poetry run python -m vlarl_launcher.cli --help
    ```

### 🚀 Usage

`vlarl-launcher` provides command-line utilities to run RL training with different algorithms and policies. The codebase supports both local training with WebSocket-based agent communication and distributed training with Ray.

#### Available Components

**Algorithms:**
- `dummy` - Dummy algorithm for testing and debugging
- `dppo` - DPPO (Diffusion-based Policy Optimization) algorithm
- `dppo-dist` - Distributed DPPO (experimental)

**Policies:**
- `dummy-policy` - Dummy policy that outputs random actions (for testing)
- `dppo-policy` - DPPO policy (requires `vlarl-infra[dppo]` and checkpoint)
- `pi0-policy` - PI0 policy (OpenPI) (requires checkpoint)

#### Quick Start: Testing with Dummy Components

To test the agent-server connection with dummy algorithm and policy:

```bash
# Terminal 1: Start the server
python -m vlarl_launcher.cli dummy default dummy-policy default

# Terminal 2: Start the environment client (from vlarl-infra)
# The server will listen on localhost:8000 by default
```

**Expected output from server:**
```
12:34:56|INFO|vlarl_launcher version: X.X.X
12:34:56|INFO|Algorithm: dummy, Config: DummyAlgoConfig(...)
12:34:56|INFO|Policy: dummy-policy, Config: DummyPolicyConfig(...)
12:34:56|INFO|WebSocket server listening on 0.0.0.0:8000
```

#### Training DPPO with DPPO Policy

```bash
# Single GPU training with WebSocket server
python -m vlarl_launcher.cli dppo hopper dppo-policy default \
  --exp_name my_dppo_exp \
  --track.enabled true

# With custom learning rates and batch size
python -m vlarl_launcher.cli dppo hopper dppo-policy default \
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
python -m vlarl_launcher.cli dppo hopper pi0-policy default \
  --policy.checkpoint_path /path/to/pi0/checkpoint \
  --exp_name pi0_dppo_exp \
  --track.enabled true

# With custom denoising steps
python -m vlarl_launcher.cli dppo hopper pi0-policy default \
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
python -m vlarl_launcher.cli_ray dppo hopper dppo-policy default \
  --num_ddp_gpus 4 \
  --exp_name dppo_dist_4gpu

# All available GPUs
python -m vlarl_launcher.cli_ray dppo hopper dppo-policy default \
  --exp_name dppo_dist_all_gpus

# With specific inference GPU
python -m vlarl_launcher.cli_ray dppo hopper dppo-policy default \
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
python -m vlarl_launcher.cli dppo hopper dppo-policy default \
  --exp_name my_dppo_exp \
  --resume true
```

#### Viewing All Available Configurations

To see all registered configurations:

```bash
# This will show the nested menu structure
python -m vlarl_launcher.cli --help
```

Output will show available combinations of algorithms and policies with their variants.