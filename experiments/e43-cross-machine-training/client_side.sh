#!/usr/bin/env bash
# E43's env clients for the cross arm, on the second machine: a Windows laptop
# that reaches guangzhao over Tailscale. The same command line as the local
# arm's clients, with guangzhao's address in place of 127.0.0.1.
#
#   bash client_side.sh STEPS [OUT_DIR]
#
# PLUGRL_CLIENT_DIR is an env-client checkout at guangzhao's commit, with the
# mujoco extra and guangzhao's mujoco and gymnasium versions.
set -uo pipefail

STEPS="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:-$HERE/results/cross-clients}"
mkdir -p "$OUT"
HOST="${SERVER_HOST:-100.75.226.89}"
PORT_BASE="${PORT_BASE:-9790}"
SEEDS="${SEEDS:-0 1 2}"
CLIENT_DIR="${PLUGRL_CLIENT_DIR:?set PLUGRL_CLIENT_DIR to the env-client checkout}"
CLIENT_PY="$CLIENT_DIR/.venv/Scripts/python.exe"
[ -x "$CLIENT_PY" ] || CLIENT_PY="$CLIENT_DIR/.venv/bin/python"

echo "client:  $CLIENT_PY"
echo "server:  $HOST ports $PORT_BASE+seed"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host "$HOST" \
      --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed "$seed" \
      --exp-name "e43-cross-seed$seed" \
      > "$OUT/client-seed$seed.log" 2>&1)
  echo "seed $seed client rc=$? at $(date '+%T')"
}

PIDS=()
for seed in $SEEDS; do
  run_seed "$seed" &
  PIDS+=($!)
  sleep 5
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
date '+end   %F %T'
echo "CLIENTS_DONE"
