#!/usr/bin/env bash
# E25 on qz103: pi0.5 trained by DPPO for two iterations, then evaluated.
#
#   setsid nohup bash e25/run.sh > e25/run.out 2>&1 < /dev/null &
#
# Server on cards 0 and 1, clients on card 2 (cell.sh); then the last
# checkpoint evaluated on cards 0 and 1 with E14's harness (eval.sh), and its
# movement from the base per module group (movement.py). See PROTOCOL.md.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e25
LOG=$E/run.log
CELL=dppo-libero
mkdir -p "$E/results"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "code: $(cat $R/plugrl-server-main/COMMIT)"
ALLOW_SIBLINGS=1 BATCH=8 PORT=8590 bash "$E/cell.sh" "$CELL" 2 4096 > "$E/$CELL.out" 2>&1
log "END train rc=$?"
tee -a "$LOG" < "$E/$CELL.out"

ROOT="$E/runs/$CELL/ck/dppo/pi0-policy/$CELL"
LAST=$(ls -1 "$ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
log "checkpoints: $(ls -1 "$ROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ') - evaluating ${LAST:-none}"

"$R/venv/bin/python" "$E/movement.py" "$R/e14/base-statedict/model.safetensors" \
  "$ROOT/$LAST/model.safetensors" > "$E/results/movement.txt" 2>&1
tee -a "$LOG" < "$E/results/movement.txt"

ALLOW_SIBLINGS=1 EVAL_SRV_GPU=0 EVAL_CLI_GPU=1 \
  bash "$E/eval.sh" e25-dppo-iter2 dppo-iter2 8 50 8591 21600 "$ROOT/$LAST" > "$E/eval-dppo-iter2.out" 2>&1
log "END eval rc=$?"
tee -a "$LOG" < "$E/results/stageC_eval.tsv"
log "E25_DONE"
