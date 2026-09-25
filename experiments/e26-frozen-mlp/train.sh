#!/usr/bin/env bash
# Does a second learn step fit once the optimizer's state is on another card,
# and what does the transfer cost?
#
#   bash e26/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]
#
# E26: E14's training harness, unchanged but for three things - the
# server runs $R/plugrl-server-e26 (main plus #60, deployed LF) through
# PYTHONPATH, output goes under e26/, and FREEZE, when set, is passed
# as --policy.freeze-expert-params.
#
# E11 stopped at one iteration of ten: the float32 copies and Adam's moments
# do not exist until the first learn step and then have to fit beside
# everything the second one needs. Measured, that state is 3,578 MiB; moving
# it to a second card leaves the model's card with 8,046 MiB instead of
# 11,624.
#
# Two iterations is enough to answer the question, since it was the second
# that died. Ten is the experiment, and only worth starting if this passes.
set -uo pipefail

R=/home/gotham/tmp/plugrl
CELL="$1"
ITERATIONS="${2:-2}"
PORT="${3:-8161}"
MASTER_DEVICE="${4:-cuda:1}"

BUFFER_SIZE=${BUFFER_SIZE:-4096}
BATCH_SIZE=8
N_SAMPLES=4
TASK_ID=8
SUITE=libero_10
NPROC=10
GLOBAL_STEPS=$(( ITERATIONS * BUFFER_SIZE ))

SPY=$R/venv/bin/python
CPY=$R/venv-libero/bin/python
OUT=$R/e26/$CELL

CLIENT_PATTERN="[v]env-libero.*/bin/python"
SERVER_PATTERN="[p]lugrl_server.cli pi0-policy"

# The memory recorder samples by physical index, which CUDA_VISIBLE_DEVICES
# does not remap. Follow the cards the run was actually given.
_SG="${SRV_GPUS:-0,1}"
export MEM_G0="${_SG%%,*}"
export MEM_G1="${_SG##*,}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

alive() {
  local s
  s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')
  [ -n "$s" ] && [ "${s#Z}" = "$s" ]
}

kill_group() {
  local pgid="$1"
  kill -TERM -- "-$pgid" 2>/dev/null
  for _ in $(seq 30); do
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
    ps -o pid,etimes,args -p "$pids" | cut -c1-120 | head -5
    return 1
  fi
  return 0
}

log "cell=$CELL iterations=$ITERATIONS master_device=$MASTER_DEVICE lr=${LR:-1e-5} updates=${UPDATES:-4} clip=${CLIP_EPS:-0.05} drift=${DRIFT:-0} warmup=${WARMUP:-0} srv_gpus=${SRV_GPUS:-0,1} cli_gpu=${CLI_GPU:-2} buffer=$BUFFER_SIZE batch=$BATCH_SIZE"

if [ "${ALLOW_SIBLINGS:-0}" = 1 ]; then
  log "ALLOW_SIBLINGS=1: not checking for other clients or servers"
elif ! report_strays "refusing to start, clients" "$CLIENT_PATTERN" \
  || ! report_strays "refusing to start, servers" "$SERVER_PATTERN"; then
  exit 2
fi

rm -rf "$OUT"; mkdir -p "$OUT"

MASTER_FLAG=()
[ "$MASTER_DEVICE" != "none" ] && MASTER_FLAG=(--algo.master-weights-device "$MASTER_DEVICE")

# Only passed when set, so the harness still runs against a server that does
# not have the flag at all.
OPT_FLAGS=()
[ -n "${WARMUP:-}" ] && [ "${WARMUP:-0}" != 0 ] \
  && OPT_FLAGS+=(--algo.critic-warmup-iterations "$WARMUP")
[ -n "${DRIFT:-}" ] && [ "${DRIFT:-0}" != 0 ] \
  && OPT_FLAGS+=(--algo.max-policy-drift "$DRIFT")
[ -n "${FREEZE:-}" ] \
  && OPT_FLAGS+=(--policy.freeze-expert-params "$FREEZE")

(
  cd "$R/plugrl-server-e26" && exec setsid env \
    PYTHONPATH="$R/plugrl-server-e26/src" \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" \
    CUDA_VISIBLE_DEVICES=${SRV_GPUS:-0,1} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$SPY" -m plugrl_server.cli pi0-policy default fpo default \
      --policy.name pi05_libero \
      --policy.checkpoint-path "$R/ckpt/pi05_libero" \
      --policy.device cuda:0 \
      --algo.learning-rate ${LR:-1e-5} \
      --algo.batch-size "$BATCH_SIZE" \
      --algo.num-updates-per-batch ${UPDATES:-4} \
      --algo.n-samples-per-action "$N_SAMPLES" \
      --algo.buffer-size "$BUFFER_SIZE" \
      --algo.clipping-epsilon ${CLIP_EPS:-0.05} \
      --algo.global-steps "$GLOBAL_STEPS" \
      --algo.save-interval 1 \
      "${MASTER_FLAG[@]}" \
      "${OPT_FLAGS[@]}" \
      --seed 7 \
      --port "$PORT" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT/ck" --exp-name "$CELL" --overwrite
) > "$OUT/server.log" 2>&1 &
SPID=$!

# Peak memory on both cards, sampled from outside: nothing in the server
# records it, and which card holds what is the point here.
(
  exec setsid bash -c 'while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $MEM_G0),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $MEM_G1)"; sleep 10; done'
) > "$OUT/gpu_mem.csv" 2>/dev/null &
MPID=$!

teardown() {
  kill_group "$SPID"
  [ -n "${CPID:-}" ] && kill_group "$CPID"
  kill_group "$MPID"
  if report_strays "after teardown, clients" "$CLIENT_PATTERN" \
    && report_strays "after teardown, servers" "$SERVER_PATTERN"; then
    log "teardown clean"
  else
    log "TEARDOWN_NOT_CLEAN"
  fi
}

for _ in $(seq 1800); do
  grep -q "is listening" "$OUT/server.log" 2>/dev/null && break
  alive "$SPID" || break
  sleep 1
done
if ! grep -q "is listening" "$OUT/server.log" 2>/dev/null; then
  log "server never listened"; tail -15 "$OUT/server.log"; teardown; exit 1
fi
log "server listening"

T0=$(date +%s)
(
  cd "$OUT" && exec setsid env \
    LIBERO_CONFIG_PATH="$R/.libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=${CLI_GPU:-2} MUJOCO_EGL_DEVICE_ID=${CLI_GPU:-2} \
    XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    timeout 43200 "$CPY" -m plugrl_env_client.cli libero-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" \
      --num-envs 1 --num-procs "$NPROC" --num-episodes 1000000000 \
      --env.task-suite-name "$SUITE" \
      --env.task-id "$TASK_ID" \
      --env.randomize-initial-state \
      --runner.replan-steps 5 --runner.seed 7 \
      --recorder.no-thread0-only \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

SERVER_DIED=0
while alive "$CPID"; do
  if ! alive "$SPID"; then
    log "server exited"
    SERVER_DIED=1
    for _ in $(seq 60); do alive "$CPID" || break; sleep 1; done
    kill_group "$CPID"
    break
  fi
  sleep 15
done
wait "$CPID"
CEXIT=$?
WALL=$(( $(date +%s) - T0 ))
log "clients exit $CEXIT after ${WALL}s (server_died=$SERVER_DIED)"

teardown

log "=== learn steps seen ==="
grep -cE "learn|Learn" "$OUT/server.log" 2>/dev/null | head -1
log "=== out of memory? ==="
grep -c "OutOfMemoryError" "$OUT/server.log" 2>/dev/null | head -1
log "=== peak memory, GPU0 and GPU1 ==="
awk -F, 'NR>0 {if ($2>m0) m0=$2; if ($3>m1) m1=$3} END {print "gpu0_peak_mib=" m0 "  gpu1_peak_mib=" m1}' "$OUT/gpu_mem.csv" 2>/dev/null
log "=== checkpoints ==="
find "$OUT/ck" -name model.safetensors -printf '%TH:%TM %s %p\n' 2>/dev/null | sort | tail -5
log "E26_TRAIN_DONE $CELL"
