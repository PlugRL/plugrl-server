#!/usr/bin/env bash
# Build a CUDA-free copy of the LIBERO env client environment.
#
# The question this answers: how much of the 6.8 GB env client is there only
# because torch arrived with CUDA? The environment side renders and steps a
# simulator; it runs no model. If the CUDA half can go, the env client can be
# installed on a machine that cannot hold a training stack.
#
# venv-libero itself is E11's recorded environment and is never touched. This
# copies it, swaps the CUDA torch for the CPU build of the same version, drops
# what is then orphaned, and measures both ends.
set -uo pipefail

R=/home/gotham/tmp/plugrl
SRC=$R/venv-libero
DST=$R/venv-libero-cpu
UV=$HOME/.local/bin/uv
PY=$DST/bin/python
LOG=$R/cpu-venv-build.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

: > "$LOG"
log "source $SRC"
du -sh "$SRC" | tee -a "$LOG"

rm -rf "$DST"
log "copying, 35k files"
cp -a "$SRC" "$DST" || { log "copy failed"; exit 1; }
log "copied"
du -sh "$DST" | tee -a "$LOG"

# The CPU wheel of the same version, so nothing else has to move.
log "installing torch 2.14.0+cpu and torchvision 0.29.0+cpu"
"$UV" pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.14.0+cpu" "torchvision==0.29.0+cpu" >> "$LOG" 2>&1
log "torch install exit $?"

# Whatever CUDA packages are left are now orphaned: nothing depends on them.
NV=$("$UV" pip list --python "$PY" 2>/dev/null | awk '/^nvidia|^triton/ {print $1}' | tr '\n' ' ')
log "orphaned after the swap: $NV"
if [ -n "$NV" ]; then
  # shellcheck disable=SC2086
  "$UV" pip uninstall --python "$PY" $NV >> "$LOG" 2>&1
  log "uninstall exit $?"
fi

log "=== result ==="
du -sh "$SRC" "$DST" | tee -a "$LOG"
log "largest remaining packages:"
du -sh "$DST"/lib/python3.11/site-packages/* 2>/dev/null | sort -rh | head -6 | tee -a "$LOG"

log "=== does it still work, and is it really CUDA-free? ==="
"$PY" -c "
import torch
print('torch', torch.__version__)
print('compiled with cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
" 2>&1 | tail -4 | tee -a "$LOG"

log "BUILD_DONE"
