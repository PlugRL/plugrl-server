#!/usr/bin/env bash
# E11 Stage C, second attempt: training then the fine-tuned evaluation.
#
# The first attempt (cell C_train, 2026-09-16 18:18) completed iteration 1 and
# then ran out of GPU memory in iteration 2's collection: a learn step's blocks
# stay in PyTorch's caching allocator, and inference could not find 124 MiB.
# Fixed in plugrl-server #21; this run also asks the allocator for expandable
# segments and saves a checkpoint every iteration. PROTOCOL.md allows one
# restart, and this is it.
#
# The baseline evaluation is not repeated. It ran valid on 2026-09-16 at 17:35
# (26 of 50) and neither #20 nor #21 changes the evaluation path: post_learn is
# FPO's, which eval never calls, and the feedback wait only behaves differently
# while a learn step is in flight, which never happens during an evaluation.
#
# The first attempt's logs are kept: this one runs as cell C_train2.
set -uo pipefail

E=/home/gotham/tmp/plugrl/e11
cd "$E" || exit 1

TASK=8
PORT=8700
EPISODES=50
ITERATIONS=10
TRAIN_CELL=C_train2

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

log "training: $ITERATIONS iterations, batch_size 8, a checkpoint every iteration"
E11_FPO_BATCH_SIZE=8 bash "$E/e11_stageC_train.sh" "$TRAIN_CELL" "$TASK" "$ITERATIONS" "$PORT" 72000 \
  > "$E/$TRAIN_CELL.log" 2>&1
if ! grep -q "STAGEC_TRAIN_DONE $TRAIN_CELL" "$E/$TRAIN_CELL.log" \
  || ! grep -q "teardown clean" "$E/$TRAIN_CELL.log"; then
  log "training harness did not finish cleanly; fine-tuned evaluation not started"
  tail -5 "$E/$TRAIN_CELL.log"
  exit 1
fi

CK_ROOT="$E/runs-$TRAIN_CELL/ck/fpo/pi0-policy/$TRAIN_CELL"
LAST_STEP=$(ls "$CK_ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "$LAST_STEP" ] || [ ! -f "$CK_ROOT/$LAST_STEP/model.safetensors" ]; then
  log "no checkpoint was written; fine-tuned evaluation not started"
  exit 1
fi
if grep -q "Stopping server as the algorithm signaled to stop" "$E/runs-$TRAIN_CELL/server.log"; then
  LABEL=finetuned
  log "training reached its end; evaluating the checkpoint at step $LAST_STEP"
else
  LABEL=finetuned_incomplete
  log "training stopped early; evaluating the last saved checkpoint, step $LAST_STEP, labelled incomplete"
fi

log "fine-tuned evaluation: task $TASK, $EPISODES episodes, policy $LABEL"
bash "$E/e11_stageC_eval.sh" C_eval_finetuned "$LABEL" "$TASK" "$EPISODES" "$PORT" 28800 \
  "$CK_ROOT/$LAST_STEP" > "$E/C_eval_finetuned.log" 2>&1
if grep -q "STAGEC_EVAL_DONE C_eval_finetuned" "$E/C_eval_finetuned.log" \
  && awk -F'\t' '$1 == "C_eval_finetuned" && $13 == "true" {ok = 1} END {exit !ok}' \
     "$E/results/stageC_eval_correctness.tsv"; then
  log "fine-tuned evaluation valid: $(grep -E '^CELL' "$E/C_eval_finetuned.log")"
else
  log "fine-tuned evaluation is not valid"
  tail -5 "$E/C_eval_finetuned.log"
fi
log "STAGEC_CHAIN2_DONE"
