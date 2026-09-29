#!/usr/bin/env bash
# E43's training servers, on guangzhao: the fpo-policy · FPO · HalfCheetah
# cell of the coverage figure, E16 Phase A's command lines unchanged.
#
#   bash server_side.sh local STEPS [OUT_DIR]   # servers and env clients here
#   bash server_side.sh cross STEPS [OUT_DIR]   # servers only; the env clients
#                                               # connect from another machine
#                                               # (client_side.sh)
#
# One server per seed, as in E16. The server seed initialises the policy and
# the client seed the environment, and the two arms use the same pairs.
set -uo pipefail

ARM="$1"
STEPS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/results/$ARM}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
case "$ARM" in
  local) PORT_BASE="${PORT_BASE:-9780}" ;;
  cross) PORT_BASE="${PORT_BASE:-9790}" ;;
  *) echo "arm must be local or cross" >&2; exit 2 ;;
esac

SEEDS="${SEEDS:-0 1 2}"
BUFFER="${BUFFER:-4096}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
CLIENT_DIR="${PLUGRL_ENV_CLIENT:-$SERVER_DIR/../plugrl-env-client}"
SERVER_PY="${PLUGRL_SERVER_PYTHON:-$SERVER_DIR/../plugrl-server/.venv/bin/python}"
CLIENT_PY="${PLUGRL_CLIENT_PYTHON:-$CLIENT_DIR/.venv/bin/python}"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"

echo "arm:     $ARM  ports $PORT_BASE+seed"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
[ "$ARM" = local ] && echo "client:  $CLIENT_PY"
echo "out:     $OUT"
echo "steps:   $STEPS  seeds: $SEEDS"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && exec env PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 20 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-$ARM-seed$seed" --overwrite \
      > "$OUT/server-seed$seed.log" 2>&1) &
  local server_pid=$!

  local waited=0
  until grep -q "is listening on" "$OUT/server-seed$seed.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 180 ]; then
      echo "seed $seed: server never listened; see $OUT/server-seed$seed.log" >&2
      return 1
    fi
  done
  echo "READY seed $seed port $port at $(date '+%T')"

  if [ "$ARM" = local ]; then
    (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
        --server-port "$port" --server-host 127.0.0.1 \
        --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
        --runner.replan-steps 1 --runner.seed "$seed" \
        --exp-name "e43-local-seed$seed" \
        > "$OUT/client-seed$seed.log" 2>&1)
    echo "seed $seed client rc=$? at $(date '+%T')"
  fi
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
