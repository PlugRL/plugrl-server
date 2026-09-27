#!/usr/bin/env bash
# E23: E16 Phase A's line on Hopper-v5 instead of HalfCheetah-v5.
#
# HalfCheetah never terminates - every episode is cut at 1,000 steps - so the
# path where an episode ends because the agent fell, and GAE must not
# bootstrap past it, has never run inside a learning run on this platform.
# Hopper terminates whenever it falls. Two things differ from Phase A and
# nothing else: the environment, and the policy's dimensions (11 and 3).
#
# The client's episode budget is the step budget, not STEPS / 1000: a Hopper
# episode starts at a few dozen steps, and a count sized for 1,000-step
# episodes would end the client inside the first few iterations. The server
# ends the run at --algo.global-steps, as it does in Phase A.
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
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
mkdir -p "$OUT"

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
  echo "error: no plugrl-env-client found. Set PLUGRL_ENV_CLIENT." >&2
  exit 1
}

pick_python() {  # $1 = repo dir, $2 = override
  local repo="$1" override="${2:-}" candidate
  if [ -n "$override" ]; then echo "$override"; return; fi
  for candidate in "$repo/.venv/Scripts/python.exe" "$repo/.venv/bin/python"; do
    [ -x "$candidate" ] && { echo "$candidate"; return; }
  done
  command -v python3 || command -v python
}
SERVER_PY="$(pick_python "$SERVER_DIR" "${PLUGRL_SERVER_PYTHON:-}")"
CLIENT_PY="$(pick_python "$CLIENT_DIR" "${PLUGRL_CLIENT_PYTHON:-}")"

"$SERVER_PY" -c "import plugrl_server" 2>/dev/null || {
  echo "error: $SERVER_PY cannot import plugrl_server." >&2; exit 1; }
"$CLIENT_PY" -c "import gymnasium; gymnasium.make('Hopper-v5')" 2>/dev/null || {
  echo "error: $CLIENT_PY cannot make Hopper-v5." >&2; exit 1; }

echo "server:  $SERVER_PY"
echo "client:  $CLIENT_PY  ($CLIENT_DIR)"
echo "out:     $OUT"
echo "steps:   $STEPS  seeds: $SEEDS"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))

  # The server seed initialises the policy; the client seed initialises the
  # environment. Varying only one of them is not three seeds.
  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      --policy.obs-dim 11 --policy.action-dim 3 \
      --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 20 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "hopper-seed$seed" --overwrite \
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
      --num-envs 1 --num-episodes "$STEPS" \
      --env.name Hopper-v5 \
      --runner.replan-steps 1 --runner.seed "$seed" \
      > "$OUT/client-seed$seed.log" 2>&1)
  echo "seed $seed finished rc=$? at $(date '+%T')"
}

PIDS=()
for seed in $SEEDS; do
  run_seed "$seed" &
  PIDS+=($!)
  # Stagger the starts so three servers do not bind and import at once.
  sleep 5
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done

date '+end   %F %T'
echo "=== checkpoints written ==="
for seed in $SEEDS; do
  dir="$OUT/fpo/fpo-policy/hopper-seed$seed"
  printf 'seed %s: %s\n' "$seed" "$(ls -1 "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
done
echo "failed seeds: $FAILED"
echo E23_DONE
