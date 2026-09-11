#!/usr/bin/env bash
# E1, second pass: does the union actually INSTALL?
#
# The first pass asked whether a consistent set of versions exists. For two of
# three pairs it does, so the metadata-level claim does not hold. This pass
# asks the weaker but more practical question: can pip actually build and
# install both stacks into one environment?
#
# READING THE RESULT. A union failing is only evidence of a *conflict* if the
# environment stack installs cleanly on its own. Otherwise the failure says
# "this environment stack is hard to install", which is a different and much
# less interesting claim. The script pairs every union with its solo baseline
# and refuses to call anything a conflict without that comparison.
#
# Run this on Linux. A build failure on Windows proves nothing about the
# ecosystem.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${E1_WORK:-$HOME/.e1-install}"
RESULTS="$HERE/results-install"
PY=3.11
TIMEOUT="${E1_TIMEOUT:-1800}"

# plugrl-protocol is private, so WSL git cannot fetch it. It is installed
# from the local checkout instead; it only contributes numpy and msgpack
# and plays no part in the question this experiment asks.
PROTOCOL_LOCAL="${E1_PROTOCOL_PATH:-/mnt/d/75128/Desktop/plugrl-work/plugrl-protocol}"
DPPO_SHA="89ac4169b3c145a54ce4dc2f8f55ef6cc8980b1c"
REINFLOW_SHA="1307057af17a065f4bfd752b1000558d0514da3f"
LEROBOT_SHA="ed83cbd4f2091a3e97cdd0c48cc657020c037240"

TRAINING=(
    "websockets>=15.0.1,<16.0.0" "tyro>=0.9.32,<0.10.0" "loguru>=0.7.3,<0.8.0"
    "torch>=2.7.0,<2.8.0" "safetensors>=0.5.3,<0.6.0" "packaging>=25.0,<26.0"
    "python-dateutil>=2.9.0.post0,<3.0.0" "pyyaml>=6.0.3,<7.0.0"
    "swanlab>=0.6.10,<0.7.0" "wandb>=0.22.0,<0.23.0"
    "plugrl-protocol @ file://${PROTOCOL_LOCAL}"
    "tensordict>=0.10.0,<0.11.0" "ray>=2.51.1,<3.0.0"
    "tensorboard>=2.20.0,<3.0.0" "tqdm>=4.67.1,<5.0.0"
)

ENVCLIENT=(
    "loguru>=0.7.3,<0.8.0" "gymnasium>=1.2.0,<2.0.0" "dm-tree>=0.1.9,<0.2.0"
    "websockets>=15.0.1,<16.0.0" "msgpack>=1.1.1,<2.0.0" "tyro>=0.9.31,<0.10.0"
    "plugrl-protocol @ file://${PROTOCOL_LOCAL}"
    "pandas>=2.3.3,<3.0.0" "imageio[ffmpeg]>=2.37.2,<3.0.0"
)

D4RL=( "cython<3" "d4rl>=1.1,<2.0.0" "patchelf>=0.17.2.4,<0.18.0"
       "gymnasium-robotics[mujoco-py]>=1.4.1,<2.0.0" )
ROBOMIMIC=( "cython<3" "d4rl>=1.1,<2.0.0" "patchelf>=0.17.2.4,<0.18.0"
            "robomimic==0.3.0" "robosuite<1.5.0" "PyOpenGL==3.1.4" )
DPPO=( "dppo @ git+https://github.com/CTP314/dppo.git@${DPPO_SHA}" )
REINFLOW=( "reinflow @ git+https://github.com/CTP314/ReinFlow.git@${REINFLOW_SHA}" )
OPENPI=( "huggingface-hub<1.0" "safetensors<0.6.0"
         "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}" )

run_case() {
    local name="$1"; shift
    local log="$RESULTS/$name.log"
    local venv="$WORK/$name"
    local started outcome elapsed

    rm -rf "$venv"
    started=$(date +%s)
    {
        echo "# case:    $name"
        echo "# date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# uv:      $(uv --version 2>&1)"
        echo "# python:  $PY"
        echo "# deps:"
        printf '#   %s\n' "$@"
        echo
    } > "$log"

    if ! uv venv --python "$PY" "$venv" >> "$log" 2>&1; then
        outcome="VENV_FAILED"
    elif timeout "$TIMEOUT" \
            env VIRTUAL_ENV="$venv" uv pip install --no-progress "$@" >> "$log" 2>&1; then
        outcome="INSTALLED"
    elif [ $? -eq 124 ]; then
        outcome="TIMEOUT"
    else
        outcome="FAILED"
    fi
    elapsed=$(( $(date +%s) - started ))

    printf '%-36s %-12s %5ss\n' "$name" "$outcome" "$elapsed"
    printf '%s\t%s\t%s\n' "$name" "$outcome" "$elapsed" >> "$RESULTS/summary.tsv"
    rm -rf "$venv"
}

mkdir -p "$WORK" "$RESULTS"
: > "$RESULTS/summary.tsv"

printf '%-36s %-12s %6s\n' CASE OUTCOME TIME
printf -- '-------------------------------------------------------\n'

# Solo baselines. These decide how a union failure may be read.
run_case solo-training           "${TRAINING[@]}"
run_case solo-envclient          "${ENVCLIENT[@]}"
run_case solo-env-d4rl           "${ENVCLIENT[@]}" "${D4RL[@]}"
run_case solo-env-robomimic      "${ENVCLIENT[@]}" "${ROBOMIMIC[@]}"

printf -- '-------------------------------------------------------\n'

# Unions: what a monolithic framework installs.
run_case union-d4rl-x-reinflow   "${TRAINING[@]}" "${ENVCLIENT[@]}" "${REINFLOW[@]}" "${D4RL[@]}"
run_case union-robomimic-x-dppo  "${TRAINING[@]}" "${ENVCLIENT[@]}" "${DPPO[@]}" "${ROBOMIMIC[@]}"

printf -- '-------------------------------------------------------\n'
echo
echo "How to read this:"
echo "  solo-env-X INSTALLED + union-X FAILED  -> conflict (supports C1)"
echo "  solo-env-X FAILED                      -> that stack is simply hard to"
echo "                                            install; the union tells us"
echo "                                            nothing about conflict"
echo "  both INSTALLED                         -> no conflict at install layer"
echo
echo "logs: $RESULTS/"
