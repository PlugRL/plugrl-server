#!/usr/bin/env bash
# Can the loop run with an environment side that has no GPU at all?
#
#   bash verify_render.sh CELL PORT egl|osmesa
#
# The server is unchanged: pi0.5 on the GPU. The client is the CUDA-free
# install, rendering either through NVIDIA's EGL or through the software
# rasteriser. Pure stepping is 11x slower in software; whether that matters
# depends on how much of the loop is spent waiting for the policy, which this
# measures rather than infers.
set -uo pipefail

R=/home/gotham/tmp/plugrl
CELL="$1"
PORT="$2"
BACKEND="${3:-osmesa}"

SPY=$R/venv/bin/python
CPY=$R/venv-libero-cpu/bin/python
OUT=$R/demo/$CELL

CLIENT_PATTERN="[v]env-libero.*/bin/python"
SERVER_PATTERN="[p]lugrl_server.cli pi0-policy"

log() { echo "[$(date +%H:%M:%S)] $*"; }

alive() {
  local s
  s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')
  [ -n "$s" ] && [ "${s#Z}" = "$s" ]
}

kill_group() {
  local pgid="$1"
  kill -TERM -- "-$pgid" 2>/dev/null
  for _ in $(seq 10); do
    pgrep -g "$pgid" > /dev/null 2>&1 || return 0
    sleep 1
  done
  kill -KILL -- "-$pgid" 2>/dev/null
  sleep 1
}

report_strays() {
  local what="$1" pattern="$2" pids
  pids=$(pgrep -f "$pattern" | while read -r p; do
           [ "$(ps -o comm= -p "$p" 2>/dev/null)" = python ] && echo "$p"
         done | paste -sd, -)
  if [ -n "$pids" ]; then
    log "$what: alive"
    ps -o pid,etimes,args -p "$pids" | cut -c1-130
    return 1
  fi
  return 0
}

case "$BACKEND" in
  osmesa)
    RENDER_ENV=(
      MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
      LD_LIBRARY_PATH="$R/opt/osmesa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
      CUDA_VISIBLE_DEVICES=1
    ) ;;
  egl)
    RENDER_ENV=(
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
      CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
    ) ;;
  *) echo "backend must be egl or osmesa"; exit 2 ;;
esac

log "cell=$CELL backend=$BACKEND client=$CPY"

if ! report_strays "refusing to start, clients" "$CLIENT_PATTERN" \
  || ! report_strays "refusing to start, servers" "$SERVER_PATTERN"; then
  exit 2
fi

rm -rf "$OUT"; mkdir -p "$OUT"

(
  cd "$R/plugrl-server" && exec setsid env \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" CUDA_VISIBLE_DEVICES=0 \
    "$SPY" -m plugrl_server.cli pi0-policy default eval default \
      --policy.name pi05_libero \
      --policy.checkpoint-path "$R/ckpt/pi05_libero" \
      --policy.device cuda --port "$PORT" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT/ck" --exp-name "$CELL" --overwrite
) > "$OUT/server.log" 2>&1 &
SPID=$!

teardown() {
  kill_group "$SPID"
  [ -n "${CPID:-}" ] && kill_group "$CPID"
  report_strays "after teardown, clients" "$CLIENT_PATTERN" > /dev/null
  report_strays "after teardown, servers" "$SERVER_PATTERN" > /dev/null
  log "teardown done"
}

for _ in $(seq 1200); do
  grep -q "is listening" "$OUT/server.log" 2>/dev/null && break
  alive "$SPID" || break
  sleep 1
done
if ! grep -q "is listening" "$OUT/server.log" 2>/dev/null; then
  log "server never listened"; tail -12 "$OUT/server.log"; teardown; exit 1
fi
log "server listening"

T0=$(date +%s)
(
  cd "$OUT" && exec setsid env \
    LIBERO_CONFIG_PATH="$R/.libero" \
    XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    "${RENDER_ENV[@]}" \
    timeout 5400 "$CPY" -m plugrl_env_client.cli libero-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" \
      --num-envs 1 --num-procs 1 --num-episodes 3 \
      --env.task-suite-name libero_spatial \
      --env.task-id 0 \
      --env.no-randomize-initial-state \
      --runner.replan-steps 5 --runner.seed 7 \
      --recorder.episode-freq 1 \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

while alive "$CPID"; do
  if ! alive "$SPID"; then log "server exited early"; kill_group "$CPID"; break; fi
  sleep 5
done
wait "$CPID"
CEXIT=$?
WALL=$(( $(date +%s) - T0 ))
log "client exit $CEXIT after ${WALL}s"

teardown

log "=== summary ==="
find "$OUT" -name summary.json -exec cat {} \; 2>/dev/null | head -30
log "RENDER_LOOP_DONE $CELL $BACKEND"
