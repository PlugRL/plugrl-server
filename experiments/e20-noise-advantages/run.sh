#!/usr/bin/env bash
# E20: E16's except-critic arm, with and without reward.
#
# Same checkpoints, same restore mode, same code - this branch is cut from
# exp/e16-critic-restart - and the one change is --algo.reward-scaling 0 on
# the `zero` arm, which makes every stored reward exactly zero. With a random
# critic and no reward, the advantages are the critic's noise. See PROTOCOL.md.
#
#   bash run.sh [OUT_DIR]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/results}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9200}"
BUFFER=4096
EXTRA_STEPS=81920
FROM_STEP=327680
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
A_OUT="${A_OUT:-$SERVER_DIR/experiments/e16-critic-restart/results}"
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

"$SERVER_PY" -c "
import sys
from plugrl_server.algorithm.fpo.fpo_config import FPOAlgoConfig
sys.exit(0 if hasattr(FPOAlgoConfig(), 'restore') else 1)
" 2>/dev/null || {
  echo "error: $SERVER_PY has no --algo.restore; it is not the PR #39 tree." >&2
  exit 1; }

declare -A CKPT
nearest_checkpoint() {
  local dir="$A_OUT/fpo/fpo-policy/halfcheetah-seed$1"
  [ -d "$dir" ] || return 1
  ls -1 "$dir" 2>/dev/null | grep -E '^[0-9]+$' | awk -v want="$FROM_STEP" '
    { d = $1 - want; if (d < 0) d = -d; if (best == "" || d < best) { best = d; step = $1 } }
    END { if (best != "" && best <= 4096) print step }
  '
}
MISSING=0
for seed in $SEEDS; do
  step="$(nearest_checkpoint "$seed")"
  ck="$A_OUT/fpo/fpo-policy/halfcheetah-seed$seed/$step"
  if [ -z "$step" ] || [ ! -f "$ck/model.safetensors" ]; then
    echo "seed $seed: no Phase A checkpoint within one update of $FROM_STEP" >&2
    MISSING=$((MISSING + 1))
  else
    CKPT[$seed]="$ck"
  fi
done
[ "$MISSING" -gt 0 ] && { echo "error: $MISSING checkpoints missing." >&2; exit 1; }

echo "server:  $SERVER_PY"
echo "out:     $OUT"
date '+start %F %T'

run_arm() {   # $1 = seed, $2 = arm, $3 = reward_scaling, $4 = port
  local seed="$1" arm="$2" scaling="$3" port="$4"
  local name="$arm-seed$seed"

  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      --algo.global-steps "$EXTRA_STEPS" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 20 --algo.reward-scaling "$scaling" \
      --algo.policy-checkpoint-path "${CKPT[$seed]}" --algo.restore except-critic \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "$name" --overwrite \
      > "$OUT/server-$name.log" 2>&1 &)

  local waited=0
  until grep -q "is listening on" "$OUT/server-$name.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 180 ]; then
      echo "$name: server never listened; see $OUT/server-$name.log" >&2
      return 1
    fi
  done

  (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((EXTRA_STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed "$seed" \
      > "$OUT/client-$name.log" 2>&1)
  echo "$name finished rc=$? at $(date '+%T')"
}

FAILED=0
port=$PORT_BASE
for arm_spec in "reward 10" "zero 0"; do
  arm="${arm_spec% *}"
  scaling="${arm_spec#* }"
  echo "--- batch: $arm (reward_scaling $scaling), seeds $SEEDS ---"
  PIDS=()
  for seed in $SEEDS; do
    run_arm "$seed" "$arm" "$scaling" "$port" &
    PIDS+=($!)
    port=$((port + 1))
    sleep 5
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done
done

date '+end   %F %T'
echo "failed runs: $FAILED"
echo E20_DONE
