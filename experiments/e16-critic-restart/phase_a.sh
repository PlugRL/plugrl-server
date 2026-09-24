#!/usr/bin/env bash
# Phase A: the control, and the checkpoints Phase B starts from.
#
# E6's configuration on today's code, three seeds, stopped at 409,600 steps
# (100 updates at buffer_size 4096). E6's own published numbers are not used
# as the control: `main` has moved a long way from its cb5b369, and reading a
# 2026-09-24 run against a 2026-09-10 number would confound "the code changed"
# with "the run was restarted". This reproduces the control instead.
#
# E6's run.sh runs its seeds SERIALLY - the client sits in the foreground of
# the loop - which would take four hours here. The runs E6 actually reported
# were three concurrent servers; that launcher was not kept, so this is it.
# Three at a time is also what PROTOCOL.md fixes, because Phase B's arms have
# to meet the same CPU contention the control did.
#
#   bash phase_a.sh [STEPS] [OUT_DIR]
set -uo pipefail

STEPS="${1:-409600}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:-$HERE/results}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-8800}"
BUFFER="${BUFFER:-4096}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
mkdir -p "$OUT"

# The env client is a sibling of the *main* checkout, not of a worktree, so
# the guess E6 made resolves to nothing from here. Try the obvious places and
# say which one was taken, rather than failing with an import error.
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

# Fail now, with the reason, rather than after three empty logs.
"$SERVER_PY" -c "import plugrl_server" 2>/dev/null || {
  echo "error: $SERVER_PY cannot import plugrl_server." >&2; exit 1; }
"$CLIENT_PY" -c "import plugrl_env_client.envs.mujoco.mujoco_env" 2>/dev/null || {
  echo "error: $CLIENT_PY cannot import the MuJoCo env." >&2; exit 1; }

# The worktree's own venv is what must be used, because PR #39's --algo.restore
# does not exist in the main checkout's install. Phase A does not pass that
# flag, but Phase B does, and a Phase A run from the wrong interpreter would
# silently produce checkpoints from different code.
"$SERVER_PY" -c "
import sys
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
sys.exit(0 if hasattr(FPOAlgoConfig(), 'restore') else 1)
" 2>/dev/null || {
  echo "error: $SERVER_PY has no --algo.restore; it is not the PR #39 tree." >&2
  exit 1; }

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
      --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 20 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-seed$seed" --overwrite \
      > "$OUT/server-seed$seed.log" 2>&1 &)

  # Wait for "is listening" rather than sleeping a guessed number of seconds.
  # A bare TCP probe is not a WebSocket handshake and would put an exception
  # in the log of every run.
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
      --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
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
  dir="$OUT/fpo/fpo-policy/halfcheetah-seed$seed"
  printf 'seed %s: %s\n' "$seed" "$(ls -1 "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
done
echo "failed seeds: $FAILED"
echo PHASE_A_DONE
