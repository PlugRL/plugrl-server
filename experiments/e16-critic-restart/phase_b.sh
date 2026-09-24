#!/usr/bin/env bash
# Phase B: three arms from one checkpoint, differing by one thing each.
#
#   all            a faithful resume - model, optimizer, step, iteration.
#   model          weights only; the optimizer's moments start over.
#   except-critic  everything but critic.*, so the value head is random.
#
# The third is E14's shape: a pretrained actor, carrying its own
# normalisation, in front of a value head that has never seen it.
#
# Nine runs in three batches of three. A batch is one seed's three arms, so
# every batch holds one run of each arm and CPU contention - three processes
# on 16 cores, the same as Phase A and as E6 - cannot favour an arm.
#
#   bash phase_b.sh [PHASE_A_OUT] [OUT_DIR]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
A_OUT="${1:-$HERE/results}"
OUT="${2:-$HERE/results-b}"
case "$A_OUT" in /*) ;; *) A_OUT="$(pwd)/$A_OUT" ;; esac
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

FROM_STEP="${FROM_STEP:-327680}"
EXTRA_STEPS="${EXTRA_STEPS:-81920}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-all model except-critic}"
PORT_BASE="${PORT_BASE:-8810}"
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

# Every checkpoint must exist before any run starts. Finding this out halfway
# through batch two would mean a seed's arms ran under different conditions
# from the others.
#
# A checkpoint's directory is named for the step it was written at, and that
# is not always the round number the save interval implies: Phase A's seed 0
# wrote its first at 81921 where seeds 1 and 2 wrote theirs at 81920. So find
# the nearest rather than assume the name, and refuse anything further off
# than a single update.
declare -A CKPT
nearest_checkpoint() {   # $1 = seed; echoes the step, or nothing
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
  if [ -z "$step" ] || \
     [ ! -f "$A_OUT/fpo/fpo-policy/halfcheetah-seed$seed/$step/model.safetensors" ]; then
    echo "seed $seed: no checkpoint within one update of $FROM_STEP" >&2
    ls -1 "$A_OUT/fpo/fpo-policy/halfcheetah-seed$seed" 2>/dev/null | tr '\n' ' ' >&2
    echo >&2
    MISSING=$((MISSING + 1))
  else
    CKPT[$seed]="$step"
    [ "$step" = "$FROM_STEP" ] || echo "seed $seed: using step $step for $FROM_STEP"
  fi
done
[ "$MISSING" -gt 0 ] && { echo "error: $MISSING checkpoints missing." >&2; exit 1; }

echo "from:  $A_OUT (step $FROM_STEP)"
echo "out:   $OUT"
echo "arms:  $ARMS   seeds: $SEEDS   extra steps: $EXTRA_STEPS"
date '+start %F %T'

run_arm() {   # $1 = seed, $2 = arm, $3 = port
  local seed="$1" arm="$2" port="$3"
  local name="b-seed$seed-$arm"
  local ck="$A_OUT/fpo/fpo-policy/halfcheetah-seed$seed/${CKPT[$seed]}"

  # `all` restores global_step to FROM_STEP, so its budget is the absolute
  # step it should stop at. The other two start from zero and are given the
  # same number of new steps, which is what makes the three comparable.
  local budget="$EXTRA_STEPS"
  [ "$arm" = "all" ] && budget=$((${CKPT[$seed]} + EXTRA_STEPS))

  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      --algo.global-steps "$budget" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 20 \
      --algo.policy-checkpoint-path "$ck" --algo.restore "$arm" \
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

  # The same client seed Phase A used, for every arm. The control this is read
  # against IS Phase A, so the environment stream should differ from it as
  # little as it can. It cannot match exactly - Phase A's client is mid-stream
  # at step 327,680 and every arm's client starts fresh - and that residual
  # difference is what the `all` arm exists to measure. P2 reads it.
  (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((EXTRA_STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed "$seed" \
      > "$OUT/client-$name.log" 2>&1)
  echo "$name finished rc=$? at $(date '+%T')"
}

FAILED=0
for seed in $SEEDS; do
  echo "--- batch: seed $seed, arms $ARMS ---"
  PIDS=()
  port=$((PORT_BASE + seed * 10))
  for arm in $ARMS; do
    run_arm "$seed" "$arm" "$port" &
    PIDS+=($!)
    port=$((port + 1))
    sleep 5
  done
  for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done
done

date '+end   %F %T'
echo "failed runs: $FAILED"
echo PHASE_B_DONE
