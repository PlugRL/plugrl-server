#!/usr/bin/env bash
# E32 on qz103. See PROTOCOL.md.
#
#   setsid nohup bash e32/run.sh > e32/run.out 2>&1 < /dev/null &
#
# Phase 1, in parallel: two FPO iterations in each of four arms, each on two
# cards; the first trains only the critic, as FPO++ does. Phase 2, in
# parallel: the untrained policy and each arm's iteration-2 checkpoint
# evaluated, then every arm's movement from the base and its learn-step
# statistics. train.sh and eval.sh are E14's harnesses, derived by derive.py.
#
# Cards. The LIBERO clients' EGL context does not land on the card its id
# names; measured on this machine (EGL id -> nvidia-smi index) 0->3 1->2 2->0
# 3->1 4->7 5->6 6->4 7->5. CLI_GPU is chosen so that each arm's ten clients
# render on its own second card, which holds only the float32 master weights
# (E26: 9,070 MiB at peak), and no arm's clients land on another arm's cards.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e32
LOG=$E/run.log
mkdir -p "$E/results"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

ckpt() {   # ARM first|last: the arm's first or last checkpoint directory
  local root="$E/$1/ck/fpo/pi0-policy/$1" step pick=head
  [ "$2" = last ] && pick=tail
  step=$(ls -1 "$root" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | $pick -1)
  [ -n "$step" ] && echo "$root/$step"
}

train() {   # ARM SRV_GPUS CLI_GPU PORT FLAGS
  ALLOW_SIBLINGS=1 SRV_GPUS="$2" CLI_GPU="$3" ARM_FLAGS="$5" \
    bash "$E/train.sh" "$1" 2 "$4" cuda:1 > "$E/train-$1.out" 2>&1
  log "END train $1 rc=$?"
}

evaluate() {   # ARM SRV_GPU CLI_GPU PORT - base untrained, any other arm its last checkpoint
  local ck=()
  if [ "$1" != base ]; then
    ck=("$(ckpt "$1" last)")
    if [ -z "${ck[0]}" ]; then
      # Never let a missing checkpoint fall through to the untrained policy.
      log "END eval $1 not run: no checkpoint"
      return
    fi
  fi
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU="$2" EVAL_CLI_GPU="$3" \
    bash "$E/eval.sh" "e32-$1" "$1" 8 50 "$4" 21600 "${ck[@]}" > "$E/eval-$1.out" 2>&1
  log "END eval $1 rc=$?"
}

ARMS="control chunk persample fpopp"
WARMUP="--algo.n-critic-warmup-itrs 1"
CHUNK="--algo.output-mode u --algo.no-discretize-t-for-training --algo.cfm-loss-steps 5 --algo.cfm-loss-dims 7 --algo.cfm-loss-sum-over-steps"
PER_SAMPLE="--algo.ratio-per-sample"

log "code: $(cat $R/plugrl-server-e32/COMMIT)"
log "--- phase 1"
train control 0,1 3 8271 "$WARMUP" &
T1=$!
sleep 60
train chunk 2,3 0 8272 "$WARMUP $CHUNK" &
T2=$!
sleep 60
train persample 4,5 7 8273 "$WARMUP $PER_SAMPLE" &
T3=$!
sleep 60
train fpopp 6,7 4 8274 "$WARMUP $CHUNK $PER_SAMPLE" &
T4=$!
wait "$T1" "$T2" "$T3" "$T4"

: > "$E/results/configs.txt"
for arm in $ARMS; do
  log "checkpoints $arm: first $(ckpt $arm first || echo none), last $(ckpt $arm last || echo none)"
  { echo "## $arm"; grep -h "Algorithm: fpo" "$E/$arm/server.log" | tail -1; } >> "$E/results/configs.txt"
done

log "--- phase 2"
# Each evaluation's single client renders on its own server's card.
evaluate base 0 2 8371 &
P0=$!
evaluate control 1 3 8372 &
P1=$!
evaluate chunk 2 1 8373 &
P2=$!
evaluate persample 3 0 8374 &
P3=$!
evaluate fpopp 4 6 8375 &
P4=$!

log "--- V1: the first iteration left the actor as it was"
FIRST=()
for arm in $ARMS; do
  ck=$(ckpt $arm first) && FIRST+=("$ck/model.safetensors")
done
"$R/venv/bin/python" "$E/movement.py" "$R/e14/base-statedict/model.safetensors" \
  "${FIRST[@]}" > "$E/results/movement-iteration1.txt" 2>&1
tee -a "$LOG" < "$E/results/movement-iteration1.txt"

log "--- movement from the base after the second iteration"
LAST=()
for arm in $ARMS; do
  ck=$(ckpt $arm last) && LAST+=("$ck/model.safetensors")
done
"$R/venv/bin/python" "$E/movement.py" "$R/e14/base-statedict/model.safetensors" \
  "${LAST[@]}" > "$E/results/movement.txt" 2>&1
tee -a "$LOG" < "$E/results/movement.txt"

log "--- learn-step statistics"
for arm in $ARMS; do
  echo "## $arm"
  "$R/venv/bin/python" "$E/tb_read.py" "$E/$arm" 2>&1 | grep -v Warning
done > "$E/results/learn-stats.txt"
tee -a "$LOG" < "$E/results/learn-stats.txt"

wait "$P0" "$P1" "$P2" "$P3" "$P4"

log "=== rows ==="
tee -a "$LOG" < "$E/results/stageC_eval.tsv"
log "E32_DONE"
