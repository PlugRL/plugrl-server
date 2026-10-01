#!/usr/bin/env bash
# E46 exploration (not pre-registered): with the scheduler's 0.1 ms sleep
# replaced by a bare yield, do v1 and v2-full still differ? The server code
# is a copy of server-src with that one constant changed.
#   bash explore_sleep0.sh EXCHANGES OUT_DIR
set -uo pipefail
N="$1"
OUT="$2"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
E46="$BASE/e46"
PY="$BASE/plugrl-server/.venv/bin/python"
rm -rf "$E46/server-src-sleep0"
cp -r "$E46/server-src" "$E46/server-src-sleep0"
sed -i 's/^SCHEDULER_SLEEP_INTERVAL = 0.0001/SCHEDULER_SLEEP_INTERVAL = 0.0/' "$E46/server-src-sleep0/plugrl_server/server/websocket_agent_server.py"
grep -n '^SCHEDULER_SLEEP_INTERVAL' "$E46/server-src-sleep0/plugrl_server/server/websocket_agent_server.py"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
n=0
for spec in sleep0:v1 sleep0:v2-full sleep0:v2-full sleep0:v1 orig:v1 orig:v2-full; do
  IFS=: read -r src variant <<< "$spec"
  n=$((n + 1))
  port=$((9940 + n))
  path="$E46/server-src-sleep0:$E46/protocol-src"
  [ "$src" = orig ] && path="$E46/server-src:$E46/protocol-src"
  (cd "$OUT" && exec env PYTHONPATH="$path" "$PY" -m plugrl_server.cli \
      fpo-policy default fpo default --port "$port" --seed 0 --policy.device cpu \
      --algo.global-steps 8192 --algo.buffer-size 4096 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "sleep0-$n" --overwrite \
      > "$OUT/server-$n.log" 2>&1) &
  pid=$!
  until grep -q "is listening on" "$OUT/server-$n.log" 2>/dev/null; do sleep 1; done
  printf '%s\t' "$src"
  PYTHONPATH="$E46/protocol-src" "$PY" "$E46/bench/explore_variant_client.py" "$port" "$variant" "$N"
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
done
echo EXPLORE_SLEEP0_DONE
