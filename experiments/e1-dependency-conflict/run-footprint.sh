#!/usr/bin/env bash
# What does a rollout machine actually have to hold?
#
# E1 killed the "these cannot coexist" claim: on Linux the union installs fine.
# The difference that survived is not feasibility but footprint - how much has
# to be present on every machine that produces rollouts, and whether that
# machine needs a GPU at all.
#
# That question decides something concrete: whether a lab workstation, a CI
# runner, or a robot's onboard computer can be a rollout source. So measure it
# rather than assert it.
#
# Measured per case: wheels resolved, bytes on disk, wall-clock install time,
# and whether anything CUDA-flavoured came along.
#
# Linux only. Run through wsl-footprint.sh, which sets up the proxy and uv.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${E1_WORK:-$HOME/.e1-footprint}"
RESULTS="$HERE/results-footprint"
PY=3.11

PROTOCOL_LOCAL="${E1_PROTOCOL_PATH:-/mnt/d/75128/Desktop/plugrl-work/plugrl-protocol}"
DPPO_SHA=89ac4169b3c145a54ce4dc2f8f55ef6cc8980b1c

# egl-probe 1.0.2 ships a pre-3.5 CMakeLists; plugrl-env-client carries this
# same setting in [tool.uv].
export CMAKE_POLICY_VERSION_MINIMUM=3.5

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

measure() {
    local name="$1" role="$2"; shift 2
    local log="$RESULTS/$name.log" venv="$WORK/$name"
    local started elapsed pkgs bytes human cuda outcome

    rm -rf "$venv"
    started=$(date +%s)
    if uv venv --python "$PY" "$venv" >> "$log" 2>&1 \
       && env VIRTUAL_ENV="$venv" uv pip install --no-progress "$@" >> "$log" 2>&1
    then outcome=ok; else outcome=FAILED; fi
    elapsed=$(( $(date +%s) - started ))

    if [ "$outcome" = ok ]; then
        pkgs=$(env VIRTUAL_ENV="$venv" uv pip list 2>/dev/null | tail -n +3 | grep -c .)
        bytes=$(du -sb "$venv" 2>/dev/null | cut -f1)
        human=$(du -sh "$venv" 2>/dev/null | cut -f1)
        # nvidia-* wheels are how a CUDA-enabled torch arrives on Linux.
        cuda=$(env VIRTUAL_ENV="$venv" uv pip list 2>/dev/null \
               | grep -ciE '^(nvidia-|triton)' || true)
        [ "${cuda:-0}" -gt 0 ] && cuda="yes (${cuda} wheels)" || cuda="no"
    else
        pkgs=-; bytes=-; human=FAILED; cuda=-
    fi

    printf '%-26s %-10s %7s %9s %6ss  %s\n' \
        "$name" "$role" "${pkgs:--}" "${human:--}" "$elapsed" "$cuda"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$role" "${pkgs:--}" "${bytes:--}" "$elapsed" "$cuda" \
        >> "$RESULTS/summary.tsv"
    rm -rf "$venv"
}

mkdir -p "$WORK" "$RESULTS"
: > "$RESULTS/summary.tsv"

printf '%-26s %-10s %7s %9s %7s  %s\n' CASE ROLE PKGS DISK TIME CUDA
printf -- '--------------------------------------------------------------------------\n'

# What PlugRL asks of a machine that only runs environments.
measure envclient-bare      rollout "${ENVCLIENT[@]}"
measure envclient-d4rl      rollout "${ENVCLIENT[@]}" "${D4RL[@]}"
measure envclient-robomimic rollout "${ENVCLIENT[@]}" "${ROBOMIMIC[@]}"

printf -- '--------------------------------------------------------------------------\n'

# What a monolithic framework asks of the same machine: it colocates training
# with rollout, so every rollout host carries the training stack too.
measure monolith-d4rl       both "${TRAINING[@]}" "${ENVCLIENT[@]}" "${DPPO[@]}" "${D4RL[@]}"
measure monolith-robomimic  both "${TRAINING[@]}" "${ENVCLIENT[@]}" "${DPPO[@]}" "${ROBOMIMIC[@]}"

printf -- '--------------------------------------------------------------------------\n'

# For reference: the training side, which PlugRL needs on exactly one machine.
measure training-only       trainer "${TRAINING[@]}" "${DPPO[@]}"

printf -- '--------------------------------------------------------------------------\n'
echo "raw: $RESULTS/summary.tsv"
