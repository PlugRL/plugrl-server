#!/usr/bin/env bash
# E1: can a PlugRL training stack and a PlugRL environment stack be installed
# into one Python environment?
#
# The paper's first claim is that they cannot, and that this is a property of
# the ecosystem rather than a design preference. This script is the evidence.
#
# Method: for each case, write a pyproject.toml declaring some combination of
# dependencies and ask uv to resolve it. Resolution is metadata-only - nothing
# is downloaded or built - so a failure here is a statement about whether a
# consistent set of versions exists at all, which is the strongest form the
# claim can take.
#
# Baselines matter as much as the conflict cases. If the training stack alone
# and each environment stack alone resolve cleanly, a failure on the union
# cannot be blamed on a broken resolver or an unsatisfiable pin on one side.
#
# Usage:  bash run.sh            (all cases)
#         bash run.sh conflict   (conflict cases only)
#
# Requires: uv, and network access to PyPI and GitHub. Set HTTPS_PROXY first
# if your network needs one.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HERE/.work"
RESULTS="$HERE/results"
PY_VERSION="3.11"

# Resolved 2026-09-09. Pinned so this script keeps meaning the same thing.
PROTOCOL_SHA="c7961da3ca050717ad3a01721f00afa1ed29bf47"
DPPO_SHA="89ac4169b3c145a54ce4dc2f8f55ef6cc8980b1c"
REINFLOW_SHA="1307057af17a065f4bfd752b1000558d0514da3f"
LEROBOT_SHA="ed83cbd4f2091a3e97cdd0c48cc657020c037240"
LIBERO_SHA="f3bf9428c0d7113b0b72d1462d9552fca35fcf92"

# ---------------------------------------------------------------- dependency sets

# plugrl-server main dependencies, verbatim from its pyproject.toml.
TRAINING_STACK='
    "websockets>=15.0.1,<16.0.0",
    "tyro>=0.9.32,<0.10.0",
    "loguru>=0.7.3,<0.8.0",
    "torch>=2.7.0,<2.8.0",
    "safetensors>=0.5.3,<0.6.0",
    "packaging>=25.0,<26.0",
    "python-dateutil>=2.9.0.post0,<3.0.0",
    "pyyaml>=6.0.3,<7.0.0",
    "swanlab>=0.6.10,<0.7.0",
    "wandb>=0.22.0,<0.23.0",
    "plugrl-protocol @ git+https://github.com/PlugRL/plugrl-protocol.git@'"$PROTOCOL_SHA"'",
    "tensordict>=0.10.0,<0.11.0",
    "ray>=2.51.1,<3.0.0",
    "tensorboard>=2.20.0,<3.0.0",
    "tqdm>=4.67.1,<5.0.0",
'

# plugrl-env-client main dependencies. Note the absence of torch: this is the
# thin client the paper claims is what makes cross-stack experiments possible.
ENV_STACK='
    "loguru>=0.7.3,<0.8.0",
    "gymnasium>=1.2.0,<2.0.0",
    "dm-tree>=0.1.9,<0.2.0",
    "websockets>=15.0.1,<16.0.0",
    "msgpack>=1.1.1,<2.0.0",
    "tyro>=0.9.31,<0.10.0",
    "plugrl-protocol @ git+https://github.com/PlugRL/plugrl-protocol.git@'"$PROTOCOL_SHA"'",
    "pandas>=2.3.3,<3.0.0",
    "imageio[ffmpeg]>=2.37.2,<3.0.0",
'

POLICY_DPPO='    "dppo @ git+https://github.com/CTP314/dppo.git@'"$DPPO_SHA"'",'
POLICY_REINFLOW='    "reinflow @ git+https://github.com/CTP314/ReinFlow.git@'"$REINFLOW_SHA"'",'
POLICY_OPENPI='
    "huggingface-hub<1.0",
    "safetensors<0.6.0",
    "lerobot @ git+https://github.com/huggingface/lerobot.git@'"$LEROBOT_SHA"'",
'

ENV_D4RL='
    "cython<3",
    "d4rl>=1.1,<2.0.0",
    "patchelf>=0.17.2.4,<0.18.0",
    "gymnasium-robotics[mujoco-py]>=1.4.1,<2.0.0",
'
ENV_ROBOMIMIC='
    "cython<3",
    "d4rl>=1.1,<2.0.0",
    "patchelf>=0.17.2.4,<0.18.0",
    "robomimic==0.3.0",
    "robosuite<1.5.0",
    "PyOpenGL==3.1.4",
'
ENV_LIBERO='    "libero @ git+https://github.com/CTP314/LIBERO.git@'"$LIBERO_SHA"'",'

# ---------------------------------------------------------------- harness

write_case() {
    local name="$1" deps="$2"
    mkdir -p "$WORK/$name"
    cat > "$WORK/$name/pyproject.toml" <<EOF
[project]
name = "e1-$name"
version = "0.0.0"
requires-python = ">=$PY_VERSION,<3.12"
dependencies = [
$deps
]

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
EOF
}

run_case() {
    local name="$1" expectation="$2" deps="$3"
    write_case "$name" "$deps"

    local log="$RESULTS/$name.log"
    local started outcome elapsed
    started=$(date +%s)

    {
        echo "# case:        $name"
        echo "# expectation: $expectation"
        echo "# python:      $PY_VERSION"
        echo "# date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# uv:          $(uv --version 2>&1)"
        echo
        echo "--- pyproject.toml ---"
        cat "$WORK/$name/pyproject.toml"
        echo
        echo "--- uv lock ---"
    } > "$log"

    if (cd "$WORK/$name" && uv lock --no-progress) >> "$log" 2>&1; then
        outcome="RESOLVED"
    else
        outcome="FAILED"
    fi
    elapsed=$(( $(date +%s) - started ))

    local verdict="unexpected"
    [ "$outcome" = "$expectation" ] && verdict="as expected"

    printf '%-34s %-10s %-14s %4ss\n' "$name" "$outcome" "$verdict" "$elapsed"
    printf '%s\t%s\t%s\t%s\n' "$name" "$expectation" "$outcome" "$elapsed" >> "$RESULTS/summary.tsv"
}

# ---------------------------------------------------------------- cases

mkdir -p "$WORK" "$RESULTS"
: > "$RESULTS/summary.tsv"

only="${1:-all}"

printf '%-34s %-10s %-14s %5s\n' CASE OUTCOME VERDICT TIME
printf -- '---------------------------------------------------------------------\n'

if [ "$only" != "conflict" ]; then
    # Each side alone must resolve. Without this, a failure below proves nothing.
    run_case "baseline-training-stack"   RESOLVED "$TRAINING_STACK"
    run_case "baseline-training-dppo"    RESOLVED "$TRAINING_STACK$POLICY_DPPO"
    run_case "baseline-training-openpi"  RESOLVED "$TRAINING_STACK$POLICY_OPENPI"
    run_case "baseline-training-reinflow" RESOLVED "$TRAINING_STACK$POLICY_REINFLOW"
    run_case "baseline-env-stack"        RESOLVED "$ENV_STACK"
    run_case "baseline-env-d4rl"         RESOLVED "$ENV_STACK$ENV_D4RL"
    run_case "baseline-env-robomimic"    RESOLVED "$ENV_STACK$ENV_ROBOMIMIC"
    run_case "baseline-env-libero"       RESOLVED "$ENV_STACK$ENV_LIBERO"
    printf -- '---------------------------------------------------------------------\n'
fi

# The union of a training stack and an environment stack: everything a
# monolithic framework has to install to run that environment with that policy.
# Both main dependency sets are included, not just the environment extra - the
# extra alone understates the constraint set and lets the resolver pick, for
# instance, a gymnasium the env client would never accept.
run_case "conflict-d4rl-x-reinflow" \
    FAILED "$TRAINING_STACK$ENV_STACK$POLICY_REINFLOW$ENV_D4RL"
run_case "conflict-robomimic-x-dppo" \
    FAILED "$TRAINING_STACK$ENV_STACK$POLICY_DPPO$ENV_ROBOMIMIC"
run_case "conflict-libero-x-openpi" \
    FAILED "$TRAINING_STACK$ENV_STACK$POLICY_OPENPI$ENV_LIBERO"

printf -- '---------------------------------------------------------------------\n'
echo "logs: $RESULTS/"
echo
echo "NOTE: a RESOLVED conflict case is a real result, not a broken run. It"
echo "would mean the incompatibility is at build or import time rather than in"
echo "version metadata, which is a weaker claim and must be reported as such."
