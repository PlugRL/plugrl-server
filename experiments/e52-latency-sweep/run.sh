#!/usr/bin/env bash
# E52's runs on guangzhao, one at a time.
#   bash run.sh OUT_DIR "SEEDS" "ARMS"     # e.g. results "0 1 2" "direct 0 1 5 25"
# An arm is "direct" (the client connects to the bridge, as in E50) or a
# one-way delay in ms (the client connects through delay_proxy.py).
# The bridge listens on 8820, the proxy on 8821.
set -uo pipefail
OUT="${1:?OUT_DIR}"; SEEDS="${2:?SEEDS}"; ARMS="${3:?ARMS}"
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
PY="$HOME/zuogou/plugrl/bridges-venv/bin/python"
cd "$(dirname "$0")"
mkdir -p "$OUT"
uptime | tee -a "$OUT/load.txt"
for seed in $SEEDS; do
  for arm in $ARMS; do
    name="pendulum-$arm-seed$seed"
    echo "== $name $(date +%T)"
    if [ "$arm" = direct ]; then
      "$PY" train.py --env pendulum --arm bridge --seed "$seed" --out "$OUT/$name" \
        --port 8820 > "$OUT/$name.log" 2>&1
    else
      "$PY" delay_proxy.py --listen 8821 --upstream 8820 --delay-ms "$arm" > "$OUT/$name.proxy.log" 2>&1 &
      proxy=$!
      sleep 1
      "$PY" train.py --env pendulum --arm bridge --seed "$seed" --out "$OUT/$name" \
        --port 8820 --client-port 8821 > "$OUT/$name.log" 2>&1
      kill "$proxy" 2>/dev/null; wait "$proxy" 2>/dev/null
    fi
    tail -n 1 "$OUT/$name.log"
  done
done
uptime | tee -a "$OUT/load.txt"
echo E52_DONE
