#!/usr/bin/env bash
# Can a robomimic-class rollout machine drop CUDA?
#
# E1 round 4 measured envclient-robomimic at 7.2G with 16 nvidia wheels and
# concluded: "For environments like it, 'the environment side needs no GPU' is
# simply not true." That holds for a default install - torch ships CUDA by
# default on Linux. But the environment side renders and steps a simulator and
# runs no model, so the CUDA half may be removable.
#
# Four fresh installs, measured the way E1 measured:
#   envclient-robomimic      E1's row, rebuilt here as the local baseline
#   envclient-robomimic-cpu  the same, with the CPU build of the same torch
#   envclient-libero         LIBERO, which E1 never measured
#   envclient-libero-cpu     the same, CPU build
#
# Run on the cluster, where GitHub is unreachable, so protocol and LIBERO come
# from the local checkouts the E11 environment itself installs from. That is
# the same source, not a substitute.
set -uo pipefail

R=/home/gotham/tmp/plugrl
WORK=${FP_WORK:-$R/.fp-work}
RESULTS=${FP_RESULTS:-$R/fp-results}
UV=$HOME/.local/bin/uv
PY=3.11

PROTOCOL_LOCAL=$R/plugrl-protocol
LIBERO_LOCAL=$R/LIBERO
TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

export CMAKE_POLICY_VERSION_MINIMUM=3.5
export UV_CACHE_DIR=$R/.cache/uv
export TMPDIR=$R/.tmp

ENVCLIENT=(
    "loguru>=0.7.3,<0.8.0" "gymnasium>=1.2.0,<2.0.0" "dm-tree>=0.1.9,<0.2.0"
    "websockets>=15.0.1,<16.0.0" "msgpack>=1.1.1,<2.0.0" "tyro>=0.9.31,<0.10.0"
    "plugrl-protocol @ file://${PROTOCOL_LOCAL}"
    "pandas>=2.3.3,<3.0.0" "imageio[ffmpeg]>=2.37.2,<3.0.0"
)
ROBOMIMIC=( "cython<3" "d4rl>=1.1,<2.0.0" "patchelf>=0.17.2.4,<0.18.0"
            "robomimic==0.3.0" "robosuite<1.5.0" "PyOpenGL==3.1.4" )
LIBERO=( "libero @ file://${LIBERO_LOCAL}" )
CPU_TORCH=( "torch==2.14.0+cpu" "torchvision==0.29.0+cpu" )

measure() {
    local name="$1" cpu="$2"; shift 2
    local log="$RESULTS/$name.log" venv="$WORK/$name"
    local started elapsed pkgs bytes human cuda outcome
    local extra=()

    if [ "$cpu" = cpu ]; then
        extra=(--extra-index-url "$TORCH_CPU_INDEX" --index-strategy unsafe-best-match)
    fi

    rm -rf "$venv"
    started=$(date +%s)
    if "$UV" venv --python "$PY" "$venv" >> "$log" 2>&1 \
       && env VIRTUAL_ENV="$venv" "$UV" pip install --no-progress "${extra[@]}" "$@" >> "$log" 2>&1
    then outcome=ok; else outcome=FAILED; fi
    elapsed=$(( $(date +%s) - started ))

    if [ "$outcome" = ok ]; then
        pkgs=$(env VIRTUAL_ENV="$venv" "$UV" pip list 2>/dev/null | tail -n +3 | grep -c .)
        bytes=$(du -sb "$venv" 2>/dev/null | cut -f1)
        human=$(du -sh "$venv" 2>/dev/null | cut -f1)
        cuda=$(env VIRTUAL_ENV="$venv" "$UV" pip list 2>/dev/null \
               | grep -ciE '^(nvidia-|triton)' || true)
        if [ "${cuda:-0}" -gt 0 ]; then cuda="yes (${cuda} wheels)"; else cuda="no"; fi
        # A smaller install that cannot load is not a result.
        if ! "$venv/bin/python" -c "import robosuite" >> "$log" 2>&1; then
            outcome="IMPORT FAILED"
        fi
    else
        pkgs=-; bytes=-; human=FAILED; cuda=-
    fi

    printf '%-26s %7s %9s %7ss  %-16s %s\n' \
        "$name" "${pkgs:--}" "${human:--}" "$elapsed" "$cuda" "$outcome"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "${pkgs:--}" "${bytes:--}" "$elapsed" "$cuda" "$outcome" \
        >> "$RESULTS/summary.tsv"
    rm -rf "$venv"
}

mkdir -p "$WORK" "$RESULTS" "$TMPDIR"
: > "$RESULTS/summary.tsv"

printf '%-26s %7s %9s %8s  %-16s %s\n' CASE PKGS DISK TIME CUDA IMPORTS
printf -- '-------------------------------------------------------------------------------\n'

measure envclient-robomimic     gpu "${ENVCLIENT[@]}" "${ROBOMIMIC[@]}"
measure envclient-robomimic-cpu cpu "${ENVCLIENT[@]}" "${ROBOMIMIC[@]}" "${CPU_TORCH[@]}"
measure envclient-libero        gpu "${ENVCLIENT[@]}" "${ROBOMIMIC[@]}" "${LIBERO[@]}"
measure envclient-libero-cpu    cpu "${ENVCLIENT[@]}" "${ROBOMIMIC[@]}" "${LIBERO[@]}" "${CPU_TORCH[@]}"

echo
echo "rows: $RESULTS/summary.tsv"
echo "FOOTPRINT_DONE"
