#!/usr/bin/env bash
# One iteration under one changed knob, then evaluate it.
#
#   CELL=a-updates1 PORT=8191 EVAL_PORT=8291 SRV_GPUS=2,3 CLI_GPU=3 UPDATES=1 \
#     setsid nohup bash e14/arm.sh > e14/a-updates1-run.out 2>&1 &
#
# Everything not named in the environment is E14's configuration. The question
# each arm asks is the same: does bounding how far one iteration travels keep
# the policy alive? E14's own first iteration moved the action expert 1.5% and
# the output projections 2.4% and evaluated 0 of 50, against a baseline of 29.
#
# Arms run side by side. Each client is launched with its own --server-port,
# so they cannot reach each other's server; ALLOW_SIBLINGS tells the harness
# its stray check cannot tell a sibling from a leftover here.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
CELL="${CELL:?CELL is required}"
PORT="${PORT:?PORT is required}"
EVAL_PORT="${EVAL_PORT:?EVAL_PORT is required}"
CKROOT=$E/$CELL/ck/fpo/pi0-policy/$CELL
LOG=$E/$CELL.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "arm $CELL: lr=${LR:-1e-5} updates=${UPDATES:-4} clip=${CLIP_EPS:-0.05} gpus=${SRV_GPUS:-?}/${CLI_GPU:-?}"

ALLOW_SIBLINGS=1 bash "$R/e14_train.sh" "$CELL" 1 "$PORT" cuda:1 > "$E/$CELL.out" 2>&1
log "training wrapper exited rc=$?"

log "--- what the training metrics say ---"
"$R/venv/bin/python" "$E/read_tb.py" "$CKROOT/tensorboard" 2>/dev/null \
  | grep -A3 -E 'rollout/success|advantages_raw_std|policy_ratio_mean|clipped_ratio_mean|initial_cfm_loss_mean' \
  | tee -a "$LOG"

mapfile -t STEPS < <(ls -1 "$CKROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
if [ "${#STEPS[@]}" -lt 1 ]; then
  log "NO CHECKPOINT"
  tail -20 "$E/$CELL.out" | tee -a "$LOG"
  log "ARM_ABORTED $CELL"
  exit 1
fi

log "START eval-$CELL step=${STEPS[0]}"
ALLOW_SIBLINGS=1 EVAL_SRV_GPU=${EVAL_SRV_GPU:-0} EVAL_CLI_GPU=${EVAL_CLI_GPU:-1} bash "$E/e14_eval.sh" "eval-$CELL" "$CELL" 8 50 "$EVAL_PORT" 21600 "$CKROOT/${STEPS[0]}" \
  > "$E/eval-$CELL.out" 2>&1
log "END eval-$CELL rc=$?"
grep -hE 'client exited with|STAGEC_EVAL_DONE|server never listened' \
  "$E/eval-$CELL.out" 2>/dev/null | tail -3 | tee -a "$LOG"

log "=== row ==="
grep -E "^($CELL)\b" "$E/results/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "ARM_DONE $CELL"
