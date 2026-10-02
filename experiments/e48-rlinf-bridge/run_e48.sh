#!/usr/bin/env bash
# E48's six runs, one at a time, driven from the laptop. RLinf always runs on
# guangzhao; the env clients run on guangzhao (local) or here (cross).
#
#   BRIDGE_DIR=... CLIENT_PY=... CLIENT_PATH=... bash run_e48.sh
#
# BRIDGE_DIR is a plugrl-rlinf checkout at the commit PROTOCOL.md names;
# CLIENT_PY a Python with plugrl-env-client's dependencies and mujoco (no
# torch needed); CLIENT_PATH the PYTHONPATH that puts plugrl-env-client main
# and plugrl-protocol main first (Windows: ;-separated).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/results"
REMOTE="${REMOTE:-guangzhao-memeseeks}"
GZ_RUNS="/home/guangzhao/zuogou/plugrl/e48"
GZ_SCRIPT="/home/guangzhao/zuogou/plugrl/plugrl-rlinf/examples/rlinf/run_guangzhao.sh"
: "${BRIDGE_DIR:?}" "${CLIENT_PY:?}" "${CLIENT_PATH:?}"
mkdir -p "$R"
ssh "$REMOTE" "mkdir -p $GZ_RUNS"
date '+start %F %T' | tee "$R/e48.txt"

for seed in 0 1 2; do
  echo "== local seed $seed $(date '+%T')" | tee -a "$R/e48.txt"
  ssh "$REMOTE" "bash $GZ_SCRIPT $GZ_RUNS/local-seed$seed $seed local" 2>&1 | tail -4 | tee -a "$R/e48.txt"
done

for seed in 0 1 2; do
  echo "== cross seed $seed $(date '+%T')" | tee -a "$R/e48.txt"
  { echo "before:"; tailscale ping -c 3 100.75.226.89 2>&1 | tail -1; } | tee -a "$R/e48.txt"
  ssh "$REMOTE" "(nohup bash $GZ_SCRIPT $GZ_RUNS/cross-seed$seed $seed remote > $GZ_RUNS/cross-seed$seed.out 2>&1 &)"
  sleep 5
  python "$BRIDGE_DIR/examples/rlinf/cross_clients.py" \
    --client-python "$CLIENT_PY" --pythonpath "$CLIENT_PATH" \
    --run-dir "$GZ_RUNS/cross-seed$seed" --seed "$seed" \
    --out "$R/cross-seed$seed-clients" 2>&1 | tee -a "$R/e48.txt"
  ssh "$REMOTE" "cat $GZ_RUNS/cross-seed$seed/run.txt" | tail -4 | tee -a "$R/e48.txt"
  { echo "after:"; tailscale ping -c 3 100.75.226.89 2>&1 | tail -1; } | tee -a "$R/e48.txt"
done
date '+end   %F %T' | tee -a "$R/e48.txt"
echo E48_DONE | tee -a "$R/e48.txt"
