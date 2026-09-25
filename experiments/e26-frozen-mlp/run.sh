#!/usr/bin/env bash
# E26 on qz103. See PROTOCOL.md.
#
#   setsid nohup bash e26/run.sh > e26/run.out 2>&1 < /dev/null &
#
# Phase 1, in parallel: the untrained pi0.5 evaluated on the new code (card
# 7); one FPO iteration with nothing frozen (control, cards 3-4); one with the
# action expert's MLP frozen (frozen, cards 5-6). Phase 2: the freezing check,
# then both iteration-1 checkpoints evaluated. Every setting is E14's: its
# train.sh and eval.sh are E14's harnesses, derived by derive.py.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e26
LOG=$E/run.log
mkdir -p "$E/results"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

ckpt_of() {   # the arm's first (and only) checkpoint directory
  local root="$E/$1/ck/fpo/pi0-policy/$1" step
  step=$(ls -1 "$root" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | head -1)
  [ -n "$step" ] && echo "$root/$step"
}

log "code: $(cat $R/plugrl-server-e26/COMMIT)"
log "--- phase 1"
(
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU=7 EVAL_CLI_GPU=7 \
    bash "$E/eval.sh" e26-base base 8 50 8361 21600 > "$E/eval-base.out" 2>&1
  log "END eval base rc=$?"
) &
B=$!
(
  ALLOW_SIBLINGS=1 SRV_GPUS=3,4 CLI_GPU=4 \
    bash "$E/train.sh" control 1 8261 cuda:1 > "$E/train-control.out" 2>&1
  log "END train control rc=$?"
) &
C=$!
sleep 30
(
  ALLOW_SIBLINGS=1 SRV_GPUS=5,6 CLI_GPU=6 FREEZE='\.mlp\.' \
    bash "$E/train.sh" frozen 1 8262 cuda:1 > "$E/train-frozen.out" 2>&1
  log "END train frozen rc=$?"
) &
F=$!
wait "$C" "$F"

CK_CONTROL=$(ckpt_of control)
CK_FROZEN=$(ckpt_of frozen)
log "checkpoints: control=${CK_CONTROL:-none} frozen=${CK_FROZEN:-none}"
grep -h "Froze" "$E/frozen/server.log" | tail -1 | tee -a "$LOG"

log "--- V1: did freezing hold"
"$R/venv/bin/python" "$E/check_frozen.py" "$R/e14/base-statedict/model.safetensors" \
  "$CK_CONTROL/model.safetensors" "$CK_FROZEN/model.safetensors" \
  > "$E/results/check-frozen.txt" 2>&1
tee -a "$LOG" < "$E/results/check-frozen.txt"

log "--- phase 2"
(
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU=3 EVAL_CLI_GPU=4 \
    bash "$E/eval.sh" e26-control control 8 50 8362 21600 "$CK_CONTROL" > "$E/eval-control.out" 2>&1
  log "END eval control rc=$?"
) &
P=$!
(
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU=5 EVAL_CLI_GPU=6 \
    bash "$E/eval.sh" e26-frozen frozen 8 50 8363 21600 "$CK_FROZEN" > "$E/eval-frozen.out" 2>&1
  log "END eval frozen rc=$?"
) &
Q=$!
wait "$P" "$Q" "$B"

log "=== rows ==="
tee -a "$LOG" < "$E/results/stageC_eval.tsv"
log "E26_DONE"
