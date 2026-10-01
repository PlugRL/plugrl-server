#!/usr/bin/env bash
# E46 exploration (not pre-registered): explore_variant_client.py against a
# fresh plugrl-server per run, in the order v1 v2 v2-full v2-full v2 v1.
#   bash explore_variants.sh EXCHANGES OUT_DIR
set -uo pipefail
N="$1"
OUT="$2"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
E46="$BASE/e46"
PY="$BASE/plugrl-server/.venv/bin/python"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
n=0
for variant in v1 v2 v2-full v2-full v2 v1; do
  n=$((n + 1))
  port=$((9990 + n))
  (cd "$OUT" && exec env PYTHONPATH="$E46/server-src:$E46/protocol-src" "$PY" -m plugrl_server.cli \
      fpo-policy default fpo default --port "$port" --seed 0 --policy.device cpu \
      --algo.global-steps 8192 --algo.buffer-size 4096 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "variant-$n" --overwrite \
      > "$OUT/server-$n.log" 2>&1) &
  pid=$!
  until grep -q "is listening on" "$OUT/server-$n.log" 2>/dev/null; do sleep 1; done
  PYTHONPATH="$E46/protocol-src" "$PY" "$E46/bench/explore_variant_client.py" "$port" "$variant" "$N"
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
done
echo EXPLORE_VARIANTS_DONE
