#!/usr/bin/env bash
# Wait for a card, then run both E13 cells back to back.
#
# The protocol says to wait rather than compare against a differently loaded
# machine, so this waits for real headroom instead of squeezing into what is
# left beside someone else's job. Both cells run in the same window, so they
# see the same machine as each other.
set -uo pipefail

R=/home/gotham/tmp/plugrl
LOG=$R/e13/queue.log
NEED_FREE_MIB=20000
MAX_WAIT_HOURS=12

mkdir -p "$R/e13"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

: > "$LOG"
log "waiting for a GPU with ${NEED_FREE_MIB} MiB free"

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
free_gpu=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  free_gpu=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
             | awk -v need="$NEED_FREE_MIB" -F', *' '($3 - $2) >= need {print $1; exit}')
  if [ -n "$free_gpu" ]; then
    log "GPU $free_gpu has room"
    break
  fi
  sleep 120
done

if [ -z "$free_gpu" ]; then
  log "gave up after ${MAX_WAIT_HOURS}h, no card freed"
  log "E13_QUEUE_ABANDONED"
  exit 1
fi

# Both cells use the same card, chosen once, so the two are comparable.
export E13_GPU=$free_gpu
log "running ten-egl on GPU $free_gpu"
CELL_GPU=$free_gpu bash "$R/e13_cell.sh" ten-egl 8151 egl >> "$LOG" 2>&1
log "ten-egl exited $?"

sleep 30

log "running ten-osmesa on GPU $free_gpu"
CELL_GPU=$free_gpu bash "$R/e13_cell.sh" ten-osmesa 8152 osmesa >> "$LOG" 2>&1
log "ten-osmesa exited $?"

log "E13_QUEUE_DONE"
