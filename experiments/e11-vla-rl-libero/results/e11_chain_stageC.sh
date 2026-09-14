#!/usr/bin/env bash
# E11 Stage C end to end, detached on the cluster so the connection can drop.
#
#   baseline evaluation -> training -> fine-tuned evaluation
#
# Each step runs only if the one before it is valid. The rules are NOTES.md's
# of 2026-09-14: 10 learn iterations at batch_size 8 (AMENDMENT.md), only the
# checkpoint written when training stops is evaluated, and both evaluations
# run 50 episodes on task 8 with initial states 0 to 49 in order. If training
# does not reach its end, the last saved checkpoint is evaluated and labelled
# incomplete.
set -uo pipefail

E=/home/gotham/tmp/plugrl/e11
cd "$E" || exit 1

TASK=8
PORT=8700
EPISODES=50
ITERATIONS=10

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

eval_valid() {
  local cell="$1"
  grep -q "STAGEC_EVAL_DONE $cell" "$E/$cell.log" \
    && grep -q "teardown clean" "$E/$cell.log" \
    && awk -F'\t' -v c="$cell" '$1 == c && $13 == "true" {ok = 1} END {exit !ok}' "$E/results/stageC_eval_correctness.tsv"
}

log "baseline evaluation: task $TASK, $EPISODES episodes"
bash "$E/e11_stageC_eval.sh" C_eval_baseline baseline "$TASK" "$EPISODES" "$PORT" 28800 > "$E/C_eval_baseline.log" 2>&1
if ! eval_valid C_eval_baseline; then
  log "baseline evaluation is not valid; training not started"
  tail -5 "$E/C_eval_baseline.log"
  exit 1
fi
log "baseline evaluation valid: $(grep -E '^CELL' "$E/C_eval_baseline.log")"

log "training: $ITERATIONS iterations, batch_size 8"
E11_FPO_BATCH_SIZE=8 bash "$E/e11_stageC_train.sh" C_train "$TASK" "$ITERATIONS" "$PORT" 72000 > "$E/C_train.log" 2>&1
if ! grep -q "STAGEC_TRAIN_DONE C_train" "$E/C_train.log" || ! grep -q "teardown clean" "$E/C_train.log"; then
  log "training harness did not finish cleanly; fine-tuned evaluation not started"
  tail -5 "$E/C_train.log"
  exit 1
fi

CK_ROOT="$E/runs-C_train/ck/fpo/pi0-policy/C_train"
LAST_STEP=$(ls "$CK_ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "$LAST_STEP" ] || [ ! -f "$CK_ROOT/$LAST_STEP/model.safetensors" ]; then
  log "no checkpoint was written; fine-tuned evaluation not started"
  exit 1
fi
if grep -q "Stopping server as the algorithm signaled to stop" "$E/runs-C_train/server.log"; then
  LABEL=finetuned
  log "training reached its end; evaluating the checkpoint at step $LAST_STEP"
else
  LABEL=finetuned_incomplete
  log "training did not reach its end; evaluating the last saved checkpoint, step $LAST_STEP, labelled incomplete"
fi

log "fine-tuned evaluation: task $TASK, $EPISODES episodes, policy $LABEL"
bash "$E/e11_stageC_eval.sh" C_eval_finetuned "$LABEL" "$TASK" "$EPISODES" "$PORT" 28800 "$CK_ROOT/$LAST_STEP" > "$E/C_eval_finetuned.log" 2>&1
if eval_valid C_eval_finetuned; then
  log "fine-tuned evaluation valid: $(grep -E '^CELL' "$E/C_eval_finetuned.log")"
else
  log "fine-tuned evaluation is not valid"
  tail -5 "$E/C_eval_finetuned.log"
fi
log "STAGEC_CHAIN_DONE"
