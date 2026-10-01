#!/usr/bin/env bash
# E46 exploration (not pre-registered): plugrl-server under cProfile, driven by
# explore_variant_client.py as v1 and as v2-full, 3,000 exchanges each.
#   bash explore_profile.sh OUT_DIR
set -uo pipefail
OUT="$1"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
E46="$BASE/e46"
PY="$BASE/plugrl-server/.venv/bin/python"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
n=0
for variant in v1 v2-full; do
  n=$((n + 1))
  port=$((9996 + n))
  (cd "$OUT" && exec env PYTHONPATH="$E46/server-src:$E46/protocol-src" "$PY" -m cProfile -o "$OUT/$variant.prof" -m plugrl_server.cli \
      fpo-policy default fpo default --port "$port" --seed 0 --policy.device cpu \
      --algo.global-steps 3000 --algo.buffer-size 4096 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "profile-$variant" --overwrite \
      > "$OUT/server-$variant.log" 2>&1) &
  pid=$!
  until grep -q "is listening on" "$OUT/server-$variant.log" 2>/dev/null; do sleep 1; done
  PYTHONPATH="$E46/protocol-src" "$PY" "$E46/bench/explore_variant_client.py" "$port" "$variant" 3000
  wait "$pid"
  "$PY" -c "
import pstats
s = pstats.Stats('$OUT/$variant.prof')
s.sort_stats('tottime').print_stats(18)
" | sed -n '/ncalls/,$p' | cut -c1-150
done
echo EXPLORE_PROFILE_DONE
