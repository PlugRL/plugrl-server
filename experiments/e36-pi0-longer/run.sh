#!/usr/bin/env bash
# E36 on qz103. See PROTOCOL.md.
#
#   setsid nohup bash e36/run.sh > e36/run.out 2>&1 < /dev/null &
#
# Phase 1, in parallel: E32's fpopp for ten iterations on two server seeds,
# and E25's DPPO for ten. Phase 2, in parallel: every run's iteration-5 and
# iteration-10 checkpoints evaluated; then movements and learn statistics.
#
# Cards, with the EGL mapping measured for E32 (EGL id -> nvidia-smi index
# 0->3 1->2 2->0 3->1 4->7 5->6 6->4 7->5): each run's clients render on its
# own second card, and each evaluation's single client on its server's card.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e36
LOG=$E/run.log
mkdir -p "$E/results"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

nth_ckpt() {   # ROOT N: the N-th checkpoint directory under ROOT, by step
  local step
  step=$(ls -1 "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | sed -n "${2}p")
  [ -n "$step" ] && echo "$1/$step"
}
root_of() {    # RUN: where its checkpoints are
  case "$1" in
    dppo) echo "$E/runs/dppo/ck/dppo/pi0-policy/dppo" ;;
    *) echo "$E/$1/ck/fpo/pi0-policy/$1" ;;
  esac
}

evaluate() {   # RUN ITERATION SRV_GPU CLI_GPU PORT
  local ck
  ck=$(nth_ckpt "$(root_of "$1")" "$2")
  if [ -z "$ck" ]; then
    log "END eval $1-it$2 not run: no checkpoint"
    return
  fi
  ALLOW_SIBLINGS=1 EVAL_SRV_GPU="$3" EVAL_CLI_GPU="$4" \
    bash "$E/eval.sh" "e36-$1-it$2" "$1-it$2" 8 50 "$5" 21600 "$ck" > "$E/eval-$1-it$2.out" 2>&1
  log "END eval $1-it$2 rc=$?"
}

FPOPP="--algo.n-critic-warmup-itrs 1 --algo.output-mode u --algo.no-discretize-t-for-training --algo.cfm-loss-steps 5 --algo.cfm-loss-dims 7 --algo.cfm-loss-sum-over-steps --algo.ratio-per-sample"

log "code: $(cat $R/plugrl-server-e32/COMMIT)"
log "--- phase 1"
(
  ALLOW_SIBLINGS=1 SEED=7 SRV_GPUS=0,1 CLI_GPU=3 ARM_FLAGS="$FPOPP" \
    bash "$E/train.sh" fpopp-s7 10 8471 cuda:1 > "$E/train-fpopp-s7.out" 2>&1
  log "END train fpopp-s7 rc=$?"
) &
A=$!
sleep 60
(
  ALLOW_SIBLINGS=1 SEED=8 SRV_GPUS=2,3 CLI_GPU=0 ARM_FLAGS="$FPOPP" \
    bash "$E/train.sh" fpopp-s8 10 8472 cuda:1 > "$E/train-fpopp-s8.out" 2>&1
  log "END train fpopp-s8 rc=$?"
) &
B=$!
sleep 60
(
  ALLOW_SIBLINGS=1 SRV_GPUS=4,5 CLI_GPU=7 PORT=8473 \
    bash "$E/dppo_cell.sh" dppo 10 4096 > "$E/train-dppo.out" 2>&1
  log "END train dppo rc=$?"
) &
C=$!
wait "$A" "$B" "$C"

: > "$E/results/configs.txt"
for run in fpopp-s7 fpopp-s8; do
  { echo "## $run"; grep -h "Algorithm: fpo" "$E/$run/server.log" | tail -1; } >> "$E/results/configs.txt"
done
{ echo "## dppo"; grep -h "Algorithm: dppo" "$E/runs/dppo/server.log" | tail -1; } >> "$E/results/configs.txt"
for run in fpopp-s7 fpopp-s8 dppo; do
  log "checkpoints $run: $(ls -1 "$(root_of $run)" | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
done

log "--- phase 2"
evaluate fpopp-s7 5 0 2 8571 &
P1=$!
evaluate fpopp-s7 10 1 3 8572 &
P2=$!
evaluate fpopp-s8 5 2 1 8573 &
P3=$!
evaluate fpopp-s8 10 3 0 8574 &
P4=$!
evaluate dppo 5 4 6 8575 &
P5=$!
evaluate dppo 10 5 7 8576 &
P6=$!

log "--- movement from the base, iterations 5 and 10"
CKS=()
for run in fpopp-s7 fpopp-s8 dppo; do
  for it in 5 10; do
    ck=$(nth_ckpt "$(root_of $run)" $it) && CKS+=("$ck/model.safetensors")
  done
done
"$R/venv/bin/python" "$E/movement.py" "$R/e14/base-statedict/model.safetensors" \
  "${CKS[@]}" > "$E/results/movement.txt" 2>&1
tee -a "$LOG" < "$E/results/movement.txt"

log "--- learn-step statistics"
for run in fpopp-s7 fpopp-s8 dppo; do
  echo "## $run"
  if [ "$run" = dppo ]; then d="$E/runs/dppo"; else d="$E/$run"; fi
  "$R/venv/bin/python" "$E/tb_read.py" "$d" 2>&1 | grep -v Warning
done > "$E/results/learn-stats.txt"

wait "$P1" "$P2" "$P3" "$P4" "$P5" "$P6"

log "=== rows ==="
tee -a "$LOG" < "$E/results/stageC_eval.tsv"
log "E36_DONE"
