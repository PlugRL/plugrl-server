#!/usr/bin/env bash
# E14's five remaining evaluation cells, per PROTOCOL.md's record format.
#
#   setsid nohup bash e14/e14_evals.sh > e14/evals.out 2>&1 &
#
# eval-baseline already ran and is in results/stageC_eval.tsv at 29 of 50.
#
# The protocol asks for iter10. There is no iter10 to evaluate: the tenth
# learn step finished and the server was writing its checkpoint when the
# harness's own twelve-hour client budget killed the clients, leaving
# tmp_40960/model.safetensors 173,707,658 bytes short of its declared size
# and without config.yaml, metadata.pt or optimizer.pt. Iteration 9 at step
# 36,870 is the last complete checkpoint and stands in for it, labelled
# iter09 rather than iter10 so no row claims to be what it is not.
#
# Cards: the server on GPU 0 and the client on GPU 1, which is where
# eval-baseline ran. Another user's job sits on 0, 1 and 2 at ~10.6 GB per
# card; an evaluation is inference-only at ~8 GB, so it fits, and keeping all
# six cells on the same pair is worth more than chasing empty cards.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
CKROOT=$E/ten-iter/ck/fpo/pi0-policy/ten-iter
LOG=$E/evals.log
TASK=8
NEP=50
PORT=8182
BUDGET=7200

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {   # cell label step
  local cell="$1" label="$2" step="$3" ck="$CKROOT/$3"
  if [ ! -s "$ck/model.safetensors" ]; then
    log "SKIP $cell - no checkpoint at step $step"
    return
  fi
  log "START $cell label=$label step=$step"
  bash "$E/e14_eval.sh" "$cell" "$label" "$TASK" "$NEP" "$PORT" "$BUDGET" "$ck" > "$E/$cell.out" 2>&1
  log "END $cell rc=$?"
  grep -hE 'client exited with|STAGEC_EVAL_DONE|server never listened|TEARDOWN_NOT_CLEAN' \
    "$E/$cell.out" 2>/dev/null | tail -4 | tee -a "$LOG"
}

log "five cells to run; baseline is already recorded"

run eval-iter01        iter01        4100
run eval-iter02        iter02        8200
run eval-iter05        iter05        20480
run eval-iter09        iter09        36870
run eval-iter09-repeat iter09-repeat 36870

log "=== rows ==="
cat "$E/results/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "EVALS_DONE"
