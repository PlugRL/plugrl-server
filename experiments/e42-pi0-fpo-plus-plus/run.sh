#!/usr/bin/env bash
# E42 on qz103. See PROTOCOL.md.
#
#   setsid nohup bash e42/run.sh > e42/run.out 2>&1 < /dev/null &
#   ITERS=1 RUNS=s7 bash e42/run.sh      # a pilot: one (critic-only) iteration
#
# Phase 1, in parallel: E36's fpopp arm with the rest of FPO++'s fine-tuning
# (#91's options, as E39 ran them on square) for ten iterations on two server
# seeds. Phase 2, in parallel: each run's iteration-5 and iteration-10
# checkpoints evaluated; then movements and learn statistics. E36's run.sh
# without its DPPO run.
#
# Cards, with the EGL mapping measured for E32 (EGL id -> nvidia-smi index
# 0->3 1->2 2->0 3->1 4->7 5->6 6->4 7->5): each run's clients render on its
# own second card, and each evaluation's single client on its server's card.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e42
LOG=$E/run.log
ITERS="${ITERS:-10}"
RUNS="${RUNS:-s7 s8}"
mkdir -p "$E/results"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

nth_ckpt() {   # ROOT N: the N-th checkpoint directory under ROOT, by step
  local step
  step=$(ls -1 "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | sed -n "${2}p")
  [ -n "$step" ] && echo "$1/$step"
}
root_of() { echo "$E/$1/ck/fpo/pi0-policy/$1"; }

evaluate() {   # RUN ITERATION SRV_GPU CLI_GPU PORT
  local ck
  ck=$(nth_ckpt "$(root_of "$1")" "$2")
  if [ -z "$ck" ]; then
    log "END eval $1-it$2 not run: no checkpoint"
    return
  fi
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU="$3" EVAL_CLI_GPU="$4" \
    bash "$E/eval.sh" "e42-$1-it$2" "$1-it$2" 8 50 "$5" 21600 "$ck" > "$E/eval-$1-it$2.out" 2>&1
  log "END eval $1-it$2 rc=$?"
}

# E36's fpopp arm: FPO++'s chunk loss over the 5 executed steps and 7 used
# dimensions, velocity error, uniform times, a ratio per sample, one
# critic-only iteration.
FPOPP="--algo.n-critic-warmup-itrs 1 --algo.output-mode u --algo.no-discretize-t-for-training --algo.cfm-loss-steps 5 --algo.cfm-loss-dims 7 --algo.cfm-loss-sum-over-steps --algo.ratio-per-sample"
# The rest of FPO++'s fine-tuning, as E39 ran it on square (#91): the critic
# at 1e-4 in its own AdamW group, eps 1e-5, weight decay 1e-6, actor beta2
# 0.99, gradients clipped at 25, advantages once an iteration, raw rewards, a
# time-out as an ending, the Huber error, lambda 0.99, value loss 0.5 x MSE.
# Advantages stay normalised over the buffer: at minibatches of 8, a
# per-minibatch statistic manufactures signal (FPOAlgorithm._scale_advantage).
REST="--algo.critic-learning-rate 1e-4 --algo.adam-eps 1e-5 --algo.weight-decay 1e-6 --algo.actor-adam-beta2 0.99 --algo.max-grad-norm 25.0 --algo.no-fpo-playground-trick --algo.reward-scaling 1.0 --algo.treat-truncated-as-done --algo.cfm-loss-huber-delta 1.0 --algo.gae-lambda 0.99 --algo.value-loss-coeff 0.5"

log "code: $(cat $R/plugrl-server-e42/COMMIT)"
log "--- phase 1"
declare -A SRV=([s7]=0,1 [s8]=2,3)
declare -A CLI=([s7]=3 [s8]=0)
declare -A PORT=([s7]=8481 [s8]=8482)
declare -A SEED=([s7]=7 [s8]=8)
PIDS=()
for run in $RUNS; do
  (
    ALLOW_SIBLINGS=1 SEED=${SEED[$run]} SRV_GPUS=${SRV[$run]} CLI_GPU=${CLI[$run]} \
      CLIP_EPS=0.01 ARM_FLAGS="$FPOPP $REST" \
      bash "$E/train.sh" "fpopp-$run" "$ITERS" "${PORT[$run]}" cuda:1 > "$E/train-fpopp-$run.out" 2>&1
    log "END train fpopp-$run rc=$?"
  ) &
  PIDS+=($!)
  sleep 60
done
wait "${PIDS[@]}"

: > "$E/results/configs.txt"
for run in $RUNS; do
  { echo "## fpopp-$run"; grep -h "Algorithm: fpo" "$E/fpopp-$run/server.log" | tail -1; } >> "$E/results/configs.txt"
  log "checkpoints fpopp-$run: $(ls -1 "$(root_of fpopp-$run)" | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
done

if [ "$ITERS" -lt 10 ]; then
  log "E42_PILOT_DONE"
  exit 0
fi

log "--- phase 2"
evaluate fpopp-s7 5 0 2 8581 &
P1=$!
evaluate fpopp-s7 10 1 3 8582 &
P2=$!
evaluate fpopp-s8 5 2 1 8583 &
P3=$!
evaluate fpopp-s8 10 3 0 8584 &
P4=$!

log "--- movement from the base, iterations 5 and 10"
CKS=()
for run in fpopp-s7 fpopp-s8; do
  for it in 5 10; do
    ck=$(nth_ckpt "$(root_of $run)" $it) && CKS+=("$ck/model.safetensors")
  done
done
"$R/venv/bin/python" "$E/movement.py" "$R/e14/base-statedict/model.safetensors" \
  "${CKS[@]}" > "$E/results/movement.txt" 2>&1
tee -a "$LOG" < "$E/results/movement.txt"

log "--- learn-step statistics"
for run in fpopp-s7 fpopp-s8; do
  echo "## $run"
  "$R/venv/bin/python" "$E/tb_read.py" "$E/$run" 2>&1 | grep -v Warning
done > "$E/results/learn-stats.txt"

wait "$P1" "$P2" "$P3" "$P4"

log "=== rows ==="
tee -a "$LOG" < "$E/results/stageC_eval.tsv"
log "E42_DONE"
