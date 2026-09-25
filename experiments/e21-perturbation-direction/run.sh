#!/usr/bin/env bash
# E21 on qz103: build the arms, check them, evaluate them in two batches of four.
#
#   setsid nohup bash e21/run.sh > e21/run.out 2>&1 < /dev/null &
#
# The evaluation is E14's harness, e14/e14_eval.sh, called exactly as E15's
# arm evaluations called it; only the checkpoint and the results directory
# differ. Nothing is evaluated unless make_arms.py's checks pass.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e21
BASE=$R/e14/base-statedict/model.safetensors
FPO=$R/e14/ten-iter/ck/fpo/pi0-policy/ten-iter/4100/model.safetensors
RES=$E/results
LOG=$E/run.log
mkdir -p "$RES"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if [ ! -f "$E/arms/.built" ]; then
  log "building arms"
  if ! "$R/venv/bin/python" "$E/make_arms.py" "$BASE" "$FPO" "$E/arms" > "$RES/construction.txt" 2>&1; then
    log "construction failed its checks - nothing evaluated"
    tail -20 "$RES/construction.txt" | tee -a "$LOG"
    exit 1
  fi
  touch "$E/arms/.built"
  log "arms built, checks pass"
  sha256sum "$E"/arms/*/model.safetensors | sed "s|$E/||" > "$RES/arms.sha256"
fi

eval_one() {   # arm srv_gpu cli_gpu port
  local arm="$1" srv="$2" cli="$3" port="$4"
  log "START e21-$arm gpus=$srv/$cli port=$port"
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU="$srv" EVAL_CLI_GPU="$cli" E14_RES="$RES" \
    bash "$R/e14/e14_eval.sh" "e21-$arm" "$arm" 8 50 "$port" 21600 "$E/arms/$arm" \
    > "$E/eval-$arm.out" 2>&1
  log "END e21-$arm rc=$?"
  grep -hE 'client exited with|STAGEC_EVAL_DONE|server never listened|TEARDOWN_NOT_CLEAN' \
    "$E/eval-$arm.out" 2>/dev/null | tail -3 | tee -a "$LOG"
}

batch() {   # four arms, one per card pair
  log "--- batch: $*"
  eval_one "$1" 0 1 8391 &
  local a=$!
  eval_one "$2" 2 3 8392 &
  local b=$!
  eval_one "$3" 4 5 8393 &
  local c=$!
  eval_one "$4" 6 7 8394 &
  local d=$!
  wait "$a" "$b" "$c" "$d"
}

batch flip-s1 flip-s2 neg flip-s1-x3
batch flip-s3 flip-s1-x10 flip-s1-x30 neg-x3

log "=== rows ==="
cat "$RES/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "E21_DONE"
