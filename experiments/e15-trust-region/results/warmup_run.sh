#!/usr/bin/env bash
# Two iterations with the first one training only the value head.
#
#   setsid nohup bash e14/warmup_run.sh > e14/warmup-run.out 2>&1 &
#
# Both checkpoints are evaluated, and the first one is the control that makes
# the second readable:
#
#   iteration 1  the warmup. The actor's gradients are zeroed, so its weights
#                cannot have moved. This must come back near the baseline's
#                29 of 50. If it does not, the freeze did not work and nothing
#                below means anything.
#   iteration 2  the first policy update made with a value head that has seen
#                one iteration of returns instead of none. E14's equivalent -
#                the first update from a randomly initialised critic - scored
#                0 of 50.
#
# Everything else is E14's configuration: buffer 4096, batch 8, four updates
# per batch, four samples, learning rate 1e-5, clipping 0.05, libero_10 task
# 8, ten clients, seed 7.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
CELL=warmup1
CKROOT=$E/$CELL/ck/fpo/pi0-policy/$CELL
LOG=$E/$CELL.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "two iterations, first is a critic warmup"

ALLOW_SIBLINGS=1 WARMUP=1 SRV_GPUS=2,3 CLI_GPU=3 \
  bash "$R/e14_train.sh" "$CELL" 2 8197 cuda:1 > "$E/$CELL.out" 2>&1
log "training wrapper exited rc=$?"

log "--- metrics ---"
"$R/venv/bin/python" "$E/read_tb.py" "$CKROOT/tensorboard" 2>/dev/null \
  | grep -A3 -E 'critic_warmup|rollout/success|losses/value_loss|policy_ratio_mean' \
  | tee -a "$LOG"

mapfile -t STEPS < <(ls -1 "$CKROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
log "checkpoints=${#STEPS[@]} [${STEPS[*]:-none}]"
if [ "${#STEPS[@]}" -lt 1 ]; then
  log "NO CHECKPOINT"
  tail -20 "$E/$CELL.out" | tee -a "$LOG"
  log "WARMUP_RUN_ABORTED"
  exit 1
fi

run() {   # label step srv cli port
  log "START eval-$1 step=$2"
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU="$3" EVAL_CLI_GPU="$4" \
    bash "$E/e14_eval.sh" "eval-$1" "$1" 8 50 "$5" 21600 "$CKROOT/$2" \
    > "$E/eval-$1.out" 2>&1
  log "END eval-$1 rc=$?"
}

run warmup1-iter01 "${STEPS[0]}" 2 3 8297 &
P1=$!
if [ "${#STEPS[@]}" -ge 2 ]; then
  run warmup1-iter02 "${STEPS[1]}" 4 5 8298 &
  P2=$!
  wait "$P1" "$P2"
else
  wait "$P1"
fi

log "=== rows ==="
grep -E '^(baseline|warmup1)' "$E/results/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "WARMUP_RUN_DONE"
