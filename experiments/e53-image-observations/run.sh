#!/usr/bin/env bash
# E53's 6 registered runs on guangzhao, one at a time (PROTOCOL.md).
#   bash run.sh OUT_DIR
set -uo pipefail
OUT="${1:?OUT_DIR}"
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
PY="$HOME/zuogou/plugrl/bridges-venv/bin/python"
cd "$(dirname "$0")"
mkdir -p "$OUT"
uptime | tee -a "$OUT/load.txt"
(cd "$HOME/zuogou/plugrl/plugrl-bridges" && find src -name "*.py" | sort | xargs sha256sum) > "$OUT/bridges-src.sha256"
for seed in 0 1 2; do
  for arm in inprocess bridge; do
    name="pong-$arm-seed$seed"
    if ss -ltn | grep -q ":8830 "; then echo "port 8830 busy before $name; stopping"; exit 1; fi
    echo "== $name $(date +%T)"
    "$PY" train.py --arm "$arm" --seed "$seed" --out "$OUT/$name" --port 8830 > "$OUT/$name.log" 2>&1
    tail -n 1 "$OUT/$name.log"
  done
done
uptime | tee -a "$OUT/load.txt"
echo E53_DONE
