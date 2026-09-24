#!/usr/bin/env bash
# DPPO on the line E6 established for FPO.
#
# Everything here matches E16's Phase A - which is this experiment's FPO arm -
# except the algorithm: same policy, same environment, same buffer, same step
# budget, same three seeds, same three-at-a-time concurrency.
#
# The rest of DPPO's settings come from its shipped `cheetah` variant and are
# NOT matched to FPO's, because they do not correspond one to one. PROTOCOL.md
# names them, and registers no prediction about which algorithm scores higher
# for exactly that reason.
#
#   bash run.sh [STEPS] [OUT_DIR]
set -uo pipefail

STEPS="${1:-409600}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:-$HERE/results}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-8900}"
BUFFER="${BUFFER:-4096}"
VARIANT="${VARIANT:-cheetah}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
mkdir -p "$OUT"

# train_itrs is what the algorithm reads, and global_steps is derived from it
# as train_itrs * buffer_size. Setting --algo.global-steps directly would be
# overwritten by that. (`n_train_itr` in the cheetah variant is a dead field
# carried over from DPPO's own yaml naming; setting it does nothing.)
ITRS=$((STEPS / BUFFER))

find_client_dir() {
  local candidate
  for candidate in \
      "${PLUGRL_ENV_CLIENT:-}" \
      "$SERVER_DIR/../plugrl-env-client" \
      "$SERVER_DIR/../../../../plugrl-env-client"; do
    [ -n "$candidate" ] && [ -d "$candidate" ] && (cd "$candidate" && pwd) && return
  done
  return 1
}
CLIENT_DIR="$(find_client_dir)" || {
  echo "error: no plugrl-env-client found. Set PLUGRL_ENV_CLIENT." >&2; exit 1; }

pick_python() {
  local repo="$1" override="${2:-}" candidate
  if [ -n "$override" ]; then echo "$override"; return; fi
  for candidate in "$repo/.venv/Scripts/python.exe" "$repo/.venv/bin/python"; do
    [ -x "$candidate" ] && { echo "$candidate"; return; }
  done
  command -v python3 || command -v python
}
SERVER_PY="$(pick_python "$SERVER_DIR" "${PLUGRL_SERVER_PYTHON:-}")"
CLIENT_PY="$(pick_python "$CLIENT_DIR" "${PLUGRL_CLIENT_PYTHON:-}")"

"$CLIENT_PY" -c "import plugrl_env_client.envs.mujoco.mujoco_env" 2>/dev/null || {
  echo "error: $CLIENT_PY cannot import the MuJoCo env." >&2; exit 1; }

# The whole point of this experiment is that dppo runs at all. Check that the
# interpreter in hand is one where it is registered, rather than finding out
# from three servers that each died with a KeyError.
"$SERVER_PY" -c "
import sys
import plugrl_server.algorithm  # noqa: F401
from plugrl_server.algorithm.registration import REGISTERED_ALGORITHMS
sys.exit(0 if 'dppo' in REGISTERED_ALGORITHMS else 1)
" 2>/dev/null || {
  echo "error: dppo is not a registered algorithm for $SERVER_PY." >&2
  echo "       This needs the tree from PR #40." >&2
  exit 1; }

echo "server:  $SERVER_PY"
echo "client:  $CLIENT_PY  ($CLIENT_DIR)"
echo "out:     $OUT"
echo "algo:    dppo/$VARIANT   buffer $BUFFER x $ITRS itrs = $((BUFFER * ITRS)) steps"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))

  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default dppo "$VARIANT" \
      --port "$port" --seed "$seed" --policy.device cpu \
      --algo.buffer-size "$BUFFER" --algo.train-itrs "$ITRS" \
      --algo.save-interval 20 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-seed$seed" --overwrite \
      > "$OUT/server-seed$seed.log" 2>&1 &)

  local waited=0
  until grep -q "is listening on" "$OUT/server-seed$seed.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 180 ]; then
      echo "seed $seed: server never listened; see $OUT/server-seed$seed.log" >&2
      return 1
    fi
  done

  (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((BUFFER * ITRS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed "$seed" \
      > "$OUT/client-seed$seed.log" 2>&1)
  echo "seed $seed finished rc=$? at $(date '+%T')"
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
echo E17_DONE
