#!/usr/bin/env bash
# The rest of E14, unattended: wait for the baseline evaluation to finish, wait
# for the machine to have room, run the ten-iteration training, then evaluate
# the four checkpoints and the repeat.
#
#   setsid nohup bash e14/e14_run_rest.sh > e14/run-rest.out 2>&1 &
#
# PROTOCOL.md's stopping rules:
#
#   If the cluster's GPUs are occupied by other work, the run waits rather
#   than squeezing beside it.
#
# so this script waits, however long that takes, and picks its cards when they
# are actually free rather than naming them in advance. The first attempt died
# at 17:37 with CUDA out of memory on GPU 0, where another job held 3.82 GiB
# beside a run whose own peak is 98% of the card. PROTOCOL.md allows one
# restart for an external cause; that allowance is spent, so if this attempt
# dies the script stops for a human rather than trying again.
#
# The completion guard counts checkpoints. It does not look for the runner's
# E14_FEASIBILITY_DONE, which is printed on the way out whether the run worked
# or not.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
CELL=ten-iter
CKROOT=$E/$CELL/ck/fpo/pi0-policy/$CELL
LOG=$E/run-rest.log
TASK=8
NEP=50
PORT=8182
BUDGET=7200
FREE_MIB=500          # a card with less than this in use counts as free
NEED=3                # model card, master-copies card, clients card

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

wait_gone() {   # pattern label
  local waited=0
  while pgrep -f "$1" > /dev/null 2>&1; do
    sleep 30
    waited=$((waited + 30))
    [ $((waited % 900)) = 0 ] && log "still waiting on $2 (${waited}s)"
  done
}

free_cards() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F'[, ]+' -v lim="$FREE_MIB" '$2 < lim {print $1}' | sort -n
}

log "orchestrator started"
wait_gone '[e]14_eval_queue.sh' 'the baseline evaluation queue'
wait_gone '[v]env-libero.*/bin/python' 'env clients'
wait_gone '[p]lugrl_server.cli' 'policy servers'
sleep 30
log "our own processes are done"

# Wait for room. The protocol says wait, not squeeze, so there is no timeout
# here and no fallback that starts anyway.
prev=""
waited=0
while true; do
  cards=$(free_cards)
  n=$(printf '%s\n' "$cards" | grep -c '[0-9]')
  if [ "$n" -ge "$NEED" ]; then
    log "cards free: $(echo $cards) - taking the three highest"
    break
  fi
  if [ "$cards" != "$prev" ]; then
    log "waiting for $NEED free cards, $n free now [$(echo $cards)]"
    prev="$cards"
  fi
  waited=$((waited + 300))
  [ $((waited % 3600)) = 0 ] && log "still waiting for cards (${waited}s)"
  sleep 300
done

# Highest indices: the ones a job with a default cuda:0 reaches last.
PICK=$(printf '%s\n' "$cards" | tail -3 | tr '\n' ' ')
G1=$(echo "$PICK" | awk '{print $1}')
G2=$(echo "$PICK" | awk '{print $2}')
G3=$(echo "$PICK" | awk '{print $3}')
log "using gpus $G1,$G2 for the server and $G3 for the clients"

SRV_GPUS="$G1,$G2" CLI_GPU="$G3" bash "$R/e14_train.sh" "$CELL" 10 8181 cuda:1 \
  > "$E/$CELL.out" 2>&1
log "training wrapper exited rc=$?"

mapfile -t STEPS < <(ls -1 "$CKROOT" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
NCK=${#STEPS[@]}
log "checkpoints=$NCK [${STEPS[*]:-none}]"

if [ "$NCK" -lt 10 ]; then
  log "TRAINING_INCOMPLETE - the one allowed restart is spent, stopping for a human"
  grep -hoE 'OutOfMemoryError|CUDA out of memory|Traceback|Killed' "$E/$CELL/server.log" 2>/dev/null \
    | sort -u | tee -a "$LOG"
  tail -20 "$E/$CELL.out" | tee -a "$LOG"
  log "RUN_REST_ABORTED"
  exit 1
fi

run() {   # cell label checkpoint-dir
  local cell="$1" label="$2" ck="${3:-}"
  log "START $cell label=$label ckpt=${ck:-none}"
  bash "$E/e14_eval.sh" "$cell" "$label" "$TASK" "$NEP" "$PORT" "$BUDGET" "$ck" > "$E/$cell.out" 2>&1
  log "END $cell rc=$?"
  grep -hE 'client exited with|STAGEC_EVAL_DONE|server never listened|TEARDOWN_NOT_CLEAN' \
    "$E/$cell.out" 2>/dev/null | tail -4 | tee -a "$LOG"
}

for n in 1 2 5 10; do
  lbl=$(printf 'iter%02d' "$n")
  run "eval-$lbl" "$lbl" "$CKROOT/${STEPS[$((n - 1))]}"
done
run eval-iter10-repeat iter10-repeat "$CKROOT/${STEPS[9]}"

log "=== rows ==="
cat "$E/results/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "RUN_REST_DONE"
