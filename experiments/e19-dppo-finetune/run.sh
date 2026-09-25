#!/usr/bin/env bash
# E19: DPPO in the setting it was built for - fine-tuning a good policy.
#
# E16's except-critic arm with DPPO instead of FPO: the same three Phase A
# checkpoints at step 327,680, the actor and its observation statistics
# restored, the value head fresh, twenty iterations. See PROTOCOL.md.
#
#   bash run.sh [OUT_DIR]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/results}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9100}"
BUFFER=4096
ITRS=20
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

"$CLIENT_PY" -c "import plugrl_env_client.envs.mujoco.mujoco_env" 2>/dev/null || {
  echo "error: $CLIENT_PY cannot import the MuJoCo env." >&2; exit 1; }

# Both changes this depends on must be in the code being run: without #51
# DPPO cannot load a checkpoint at all, and without #50 it would fine-tune on
# unnormalised inputs, which is E17's defect again.
"$SERVER_PY" -c "
import sys
from plugrl_server.algorithm.dppo.dppo import DPPOAlgorithm
from plugrl_server.algorithm.dppo.dppo_config import DPPOAlgoConfig
ok = hasattr(DPPOAlgoConfig(), 'restore') and hasattr(DPPOAlgorithm, '_update_policy_obs_stats')
sys.exit(0 if ok else 1)
" 2>/dev/null || {
  echo "error: $SERVER_PY's DPPO lacks the restore (#51) or the statistics fix (#50)." >&2
  exit 1; }

# Found, not named - Phase A wrote seed 0's checkpoint at 327680 and seeds 1
# and 2's at 327681. E16's amendment 1 is the reason this is not a constant.
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
    echo "seed $seed: from $ck"
  fi
done
[ "$MISSING" -gt 0 ] && { echo "error: $MISSING checkpoints missing." >&2; exit 1; }

echo "server:  $SERVER_PY"
echo "out:     $OUT"
echo "algo:    dppo/cheetah  except-critic  buffer $BUFFER x $ITRS itrs"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))

  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default dppo cheetah \
      --port "$port" --seed "$seed" --policy.device cpu \
      --algo.buffer-size "$BUFFER" --algo.train-itrs "$ITRS" \
      --algo.save-interval 20 \
      --algo.policy-checkpoint-path "${CKPT[$seed]}" --algo.restore except-critic \
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
echo E19_DONE
