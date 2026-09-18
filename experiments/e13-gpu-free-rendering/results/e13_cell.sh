#!/usr/bin/env bash
# E13 cell: ten env clients against one server, rendering one way or the other.
#
#   bash e13_cell.sh CELL PORT egl|osmesa
#
# Per experiments/e13-gpu-free-rendering/PROTOCOL.md. The server and the
# client install are identical between cells; only the renderer differs.
# Every process writes its own summary, so the per-client timing is
# recoverable and not just the aggregate.
set -uo pipefail

R=/home/gotham/tmp/plugrl
CELL="$1"
PORT="$2"
BACKEND="${3:-osmesa}"
NPROC=10
NEP=3
# Which card the server holds, and which one the EGL cell renders on. Both
# cells take the same values, so the pair stays comparable.
SERVER_GPU="${CELL_GPU:-0}"
RENDER_GPU="${CELL_RENDER_GPU:-$SERVER_GPU}"

SPY=$R/venv/bin/python
CPY=$R/venv-libero-cpu/bin/python
OUT=$R/e13/$CELL

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
  for _ in $(seq 20); do
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
    log "$what: $(echo "$pids" | tr ',' '\n' | wc -l) alive"
    ps -o pid,etimes,args -p "$pids" | cut -c1-120 | head -6
    return 1
  fi
  return 0
}

case "$BACKEND" in
  osmesa)
    RENDER_ENV=(
      MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
      LD_LIBRARY_PATH="$R/opt/osmesa/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
      CUDA_VISIBLE_DEVICES="$RENDER_GPU"
    ) ;;
  egl)
    RENDER_ENV=(
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
      CUDA_VISIBLE_DEVICES="$RENDER_GPU" MUJOCO_EGL_DEVICE_ID="$RENDER_GPU"
    ) ;;
  *) echo "backend must be egl or osmesa"; exit 2 ;;
esac

log "cell=$CELL backend=$BACKEND procs=$NPROC episodes_per_proc=$NEP port=$PORT"
log "load at start: $(uptime | sed 's/.*load average: //')"

if ! report_strays "refusing to start, clients" "$CLIENT_PATTERN" \
  || ! report_strays "refusing to start, servers" "$SERVER_PATTERN"; then
  exit 2
fi

rm -rf "$OUT"; mkdir -p "$OUT"

(
  cd "$R/plugrl-server" && exec setsid env \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" \
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
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
  if report_strays "after teardown, clients" "$CLIENT_PATTERN" \
    && report_strays "after teardown, servers" "$SERVER_PATTERN"; then
    log "teardown clean"
  else
    log "TEARDOWN_NOT_CLEAN"
  fi
}

for _ in $(seq 1500); do
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
    timeout 7200 "$CPY" -m plugrl_env_client.cli libero-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" \
      --num-envs 1 --num-procs "$NPROC" --num-episodes "$NEP" \
      --env.task-suite-name libero_spatial \
      --env.task-id 0 \
      --env.no-randomize-initial-state \
      --runner.replan-steps 5 --runner.seed 7 \
      --recorder.episode-freq 1 --recorder.no-thread0-only \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

while alive "$CPID"; do
  if ! alive "$SPID"; then log "server exited early"; kill_group "$CPID"; break; fi
  sleep 10
done
wait "$CPID"
CEXIT=$?
WALL=$(( $(date +%s) - T0 ))
log "clients exit $CEXIT after ${WALL}s"

teardown

log "=== per process ==="
find "$OUT" -name summary.json | sort | while read -r f; do
  "$SPY" - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("timing", {})
print(
    f'proc {d.get("process_id")}\tep={d.get("completed_episodes")}'
    f'\tsr={d.get("mean_success_rate")}\tsteps={t.get("env_steps")}'
    f'\tcollect={t.get("collect_time_s", 0):.1f}s'
    f'\tstep={t.get("env_step_s", 0):.1f}s'
    f'\twait={t.get("infer_wait_s", 0):.1f}s'
    f'\tstep_frac={t.get("env_step_frac", 0):.3f}'
)
PY
done

log "CELL_WALL_SECONDS $WALL"
log "E13_CELL_DONE $CELL $BACKEND"
