#!/usr/bin/env bash
# E10 sweep, per experiments/e10-vla-forward-cost/PROTOCOL.md.
#
# Three independent process launches per cell, not three loops inside one.
# Each cell gets a wall-clock budget; a cell that exceeds it is recorded as
# not measured, with the reason, which is the protocol's stopping rule.
set -uo pipefail

ROOT=/home/gotham/tmp/plugrl
export XDG_CACHE_HOME="$ROOT/.cache" HF_HOME="$ROOT/.cache/hf" TMPDIR="$ROOT/.tmp"
# torch.compile(mode="max-autotune") caches its kernels. Pinning the cache
# makes the three launches of a cell comparable to each other and to a
# redeploy, and the first_call_ms column says what a cold one costs.
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
PY="$ROOT/venv/bin/python"
cd "$ROOT"

OUT="$ROOT/results/summary.tsv"
NOTES="$ROOT/results/not-measured.txt"
mkdir -p "$ROOT/results"
if [ ! -f "$OUT" ]; then
  printf 'policy\tdevice\tdtype\tparams\tbatch\tdenoise_steps\tlaunch\twarmup\tn\tfwd_mean_ms\tfwd_p50_ms\tfwd_p95_ms\tfwd_max_ms\tpeak_mem_mb\tfirst_call_ms\taction_shape\tvalid\n' > "$OUT"
fi
: > "$NOTES"

cell() {
  local variant="$1" device="$2" bf16="$3" budget="$4" n="$5"
  for launch in 0 1 2; do
    echo "-- $variant/$device/$bf16 launch $launch (budget ${budget}s)"
    local extra=""
    [ "$bf16" = "bf16" ] && extra="--bf16"
    if ! CUDA_VISIBLE_DEVICES=0 timeout "$budget" "$PY" e10_measure.py \
        --variant "$variant" --device "$device" $extra \
        --launch "$launch" --warmup 5 --n "$n" --out "$OUT" 2>"$ROOT/results/err-$variant-$device-$bf16-$launch.log"; then
      echo "$variant/$device/$bf16 launch $launch: NOT MEASURED (exceeded ${budget}s or failed; see err log)" >> "$NOTES"
      echo "   not measured"
    fi
  done
}

cell tiny cuda bf16 2400 30
cell base cuda bf16 2400 30
cell tiny cpu  fp32 1800 10
cell base cpu  fp32 1800 5

echo
echo "=== summary.tsv ==="
cat "$OUT"
echo
echo "=== not measured ==="
cat "$NOTES"
