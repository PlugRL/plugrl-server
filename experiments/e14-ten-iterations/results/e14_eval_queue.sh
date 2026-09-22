#!/usr/bin/env bash
# E14 evaluation queue, per experiments/e14-ten-iterations/PROTOCOL.md.
#
#   setsid nohup bash e14/e14_eval_queue.sh > e14/eval-queue.out 2>&1 &
#
# Waits for the ten-iteration training run to end, then evaluates six cells in
# sequence on one GPU pair. Each cell is e14_eval.sh, which is E11's Stage C
# harness with only its output paths changed.
#
# It refuses to spend five GPU-hours on a run that died: evaluations start only
# if training printed its completion marker, or if all ten checkpoints exist
# anyway (the wrapper died after the work was done).
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
CELL=ten-iter
CKROOT=$E/$CELL/ck/fpo/pi0-policy/$CELL
TRAINLOG=$E/$CELL.out
LOG=$E/eval-queue.log
TASK=8
NEP=50
PORT=8182
BUDGET=7200          # E11's evaluations took 1937 s and 1962 s; this is 3.7x

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "queue started, waiting for training"
while pgrep -f 'e14_train.sh' > /dev/null 2>&1; do sleep 60; done
log "training process gone"

DONE=0
grep -q 'E14_FEASIBILITY_DONE' "$TRAINLOG" 2>/dev/null && DONE=1
mapfile -t STEPS < <(ls -1 "$CKROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
NCK=${#STEPS[@]}
log "completion marker=$DONE checkpoints=$NCK [${STEPS[*]:-none}]"

if [ "$DONE" != 1 ] && [ "$NCK" -lt 10 ]; then
  log "TRAINING_DID_NOT_COMPLETE - not evaluating, tail follows"
  tail -25 "$TRAINLOG" | tee -a "$LOG"
  log "EVAL_QUEUE_ABORTED"
  exit 1
fi

for i in $(seq 0 7); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null)
  log "gpu$i used=${used:-?}MiB"
done

run() {   # cell label checkpoint-dir
  local cell="$1" label="$2" ck="${3:-}"
  log "START $cell label=$label ckpt=${ck:-none}"
  bash "$E/e14_eval.sh" "$cell" "$label" "$TASK" "$NEP" "$PORT" "$BUDGET" "$ck" \
    > "$E/$cell.out" 2>&1
  local rc=$?
  log "END $cell rc=$rc"
  grep -hE 'client exited with|STAGEC_EVAL_DONE|server never listened|TEARDOWN_NOT_CLEAN' \
    "$E/$cell.out" 2>/dev/null | tail -4 | tee -a "$LOG"
}

ck_for() { local n="$1"; local s="${STEPS[$((n - 1))]:-}"; [ -n "$s" ] && echo "$CKROOT/$s"; }

run eval-baseline baseline ""

for n in 1 2 5 10; do
  lbl=$(printf 'iter%02d' "$n")
  d=$(ck_for "$n")
  if [ -z "$d" ]; then log "SKIP $lbl - no checkpoint for iteration $n"; continue; fi
  run "eval-$lbl" "$lbl" "$d"
done

d=$(ck_for 10)
if [ -n "$d" ]; then
  run eval-iter10-repeat iter10-repeat "$d"
else
  log "SKIP iter10-repeat - no checkpoint for iteration 10"
fi

log "=== rows ==="
cat "$E/results/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "EVAL_QUEUE_DONE"
