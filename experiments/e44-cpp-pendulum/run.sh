#!/usr/bin/env bash
# E44's runs, on guangzhao: gaussian-policy · PPO on Pendulum, with the env
# client either the C++ program in this directory or plugrl-env-client
# stepping gymnasium's Pendulum-v1 (the control).
#
#   bash run.sh cpp    ITERS [OUT_DIR]
#   bash run.sh python ITERS [OUT_DIR]
#
# PPO stops after ITERS iterations (--algo.train-itrs); its step count is
# ITERS x the buffer size.
#
# One server per seed, the server seed also the client seed. Extra server
# flags (the PPO settings PROTOCOL.md fixes) come from PPO_ARGS.
set -uo pipefail

ARM="$1"
ITERS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/results/$ARM}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
case "$ARM" in
  cpp) PORT_BASE="${PORT_BASE:-9880}" ;;
  python) PORT_BASE="${PORT_BASE:-9890}" ;;
  *) echo "arm must be cpp or python" >&2; exit 2 ;;
esac

SEEDS="${SEEDS:-0 1 2}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
SERVER_PY="${PLUGRL_SERVER_PYTHON:-$SERVER_DIR/../plugrl-server/.venv/bin/python}"
CLIENT_DIR="${PLUGRL_ENV_CLIENT:?set PLUGRL_ENV_CLIENT to the env-client checkout with the classic fix}"
CLIENT_PY="${PLUGRL_CLIENT_PYTHON:-$CLIENT_DIR/.venv/bin/python}"
BINARY="${PENDULUM_CLIENT:-$HERE/pendulum_client}"
read -r -a PPO <<< "${PPO_ARGS:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"

echo "arm:     $ARM  ports $PORT_BASE+seed"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
echo "client:  $([ "$ARM" = cpp ] && echo "$BINARY" || echo "$CLIENT_PY")"
echo "ppo:     ${PPO[*]:-(defaults)}"
echo "out:     $OUT   iterations: $ITERS  seeds: $SEEDS"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && exec env PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
      gaussian-policy default ppo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      --policy.obs-dim 3 --policy.action-dim 1 --policy.action-clip 2.0 \
      --algo.train-itrs "$ITERS" "${PPO[@]}" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "pendulum-$ARM-seed$seed" --overwrite \
      > "$OUT/server-seed$seed.log" 2>&1) &
  local server_pid=$!
  local waited=0
  until grep -q "is listening on" "$OUT/server-seed$seed.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    [ "$waited" -ge 180 ] && { echo "seed $seed: server never listened" >&2; return 1; }
  done
  if [ "$ARM" = cpp ]; then
    "$BINARY" 127.0.0.1 "$port" "$seed" > "$OUT/client-seed$seed.log" 2>&1
  else
    (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli classic-v1 \
        --server-port "$port" --server-host 127.0.0.1 \
        --num-envs 1 --num-episodes 1000000 \
        --env.name Pendulum-v1 --runner.replan-steps 1 --runner.seed "$seed" \
        --exp-name "e44-python-seed$seed" \
        > "$OUT/client-seed$seed.log" 2>&1)
  fi
  echo "seed $seed client rc=$? at $(date '+%T')"
  wait "$server_pid"
  echo "seed $seed server rc=$? at $(date '+%T')"
}

PIDS=()
for seed in $SEEDS; do
  run_seed "$seed" &
  PIDS+=($!)
  sleep 5
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done
date '+end   %F %T'
echo "failed seeds: $FAILED"
echo "ARM_DONE $ARM"
