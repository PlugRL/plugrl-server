#!/usr/bin/env bash
# E25 on qz103: pi0.5 trained by DPPO on LIBERO-10 task 8, through PlugRL.
#
#   setsid nohup bash e25/cell.sh CELL ITERS BUFFER > e25/CELL.out 2>&1 < /dev/null &
#
# E14's training harness with two changes: the algorithm is DPPO's `libero`
# variant instead of FPO, and the server runs today's `main` (8812b54,
# deployed LF to $R/plugrl-server-main) through PYTHONPATH, so the copy the
# earlier experiments ran, $R/plugrl-server, is left as it was. The env
# client is the one E14 to E22 used. Server on cards 0 and 1 (policy on the
# first), ten LIBERO clients rendering on card 2, as in E14.
#
# The minibatch is 8, E14's for FPO on this policy, not the libero variant's
# 128: each minibatch runs pi0.5's VLM prefix over three images and the
# prompt, and 128 of them do not fit on a 24 GB card (E25 pilot 2). The
# variant's 16 accumulation steps, averaged since #49, still make each
# optimizer step see 128 samples.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e25
CELL="$1"
ITERS="$2"
BUFFER="$3"
OUT=$E/runs/$CELL
PORT=${PORT:-8590}
SPY=$R/venv/bin/python
CPY=$R/venv-libero/bin/python
SRC=$R/plugrl-server-main/src
mkdir -p "$OUT"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
alive() {
  local s
  s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')
  [ -n "$s" ] && [ "${s#Z}" = "$s" ]
}
kill_group() {
  kill -TERM -- "-$1" 2>/dev/null
  for _ in $(seq 10); do
    pgrep -g "$1" > /dev/null 2>&1 || return 0
    sleep 1
  done
  kill -KILL -- "-$1" 2>/dev/null
}
strays() { pgrep -f "[v]env-libero/bin/python|[p]lugrl_server.cli pi0-policy" | wc -l; }

log "cell=$CELL iters=$ITERS buffer=$BUFFER batch=${BATCH:-8} port=$PORT src=$(cat $R/plugrl-server-main/COMMIT)"
if [ "${ALLOW_SIBLINGS:-0}" = 1 ]; then
  log "ALLOW_SIBLINGS=1: not checking for other clients or servers"
elif [ "$(strays)" -ne 0 ]; then
  log "refusing to start: $(strays) client or server processes already running"
  exit 2
fi

(
  cd "$R/plugrl-server-main" && exec setsid env \
    PYTHONPATH="$SRC" \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" \
    CUDA_VISIBLE_DEVICES=${SRV_GPUS:-0,1} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$SPY" -m plugrl_server.cli pi0-policy default dppo libero \
      --policy.name pi05_libero \
      --policy.checkpoint-path "$R/ckpt/pi05_libero" \
      --policy.device cuda:0 \
      --algo.buffer-size "$BUFFER" \
      --algo.batch-size "${BATCH:-8}" \
      --algo.train-itrs "$ITERS" \
      --algo.save-interval 1 \
      --seed 7 \
      --port "$PORT" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT/ck" --exp-name "$CELL" --overwrite
) > "$OUT/server.log" 2>&1 &
SPID=$!

(
  exec setsid bash -c 'while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)"; sleep 10; done'
) > "$OUT/gpu_mem.csv" 2>/dev/null &
MPID=$!

teardown() {
  kill_group "$SPID"
  [ -n "${CPID:-}" ] && kill_group "$CPID"
  kill_group "$MPID"
  sleep 2
  # With ALLOW_SIBLINGS the count includes other experiments' processes.
  if [ "$(strays)" -eq 0 ]; then log "teardown clean"; else log "after teardown: $(strays) client or server processes on the machine"; fi
}

for _ in $(seq 1800); do
  grep -q "is listening" "$OUT/server.log" 2>/dev/null && break
  alive "$SPID" || break
  sleep 1
done
if ! grep -q "is listening" "$OUT/server.log" 2>/dev/null; then
  log "server never listened"
  tail -20 "$OUT/server.log"
  teardown
  exit 1
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
      --num-envs 1 --num-procs 10 --num-episodes 1000000000 \
      --env.task-suite-name libero_10 \
      --env.task-id 8 \
      --env.randomize-initial-state \
      --runner.replan-steps 5 --runner.seed 7 \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

# The server ends the run after ITERS learn steps; then the clients go.
while alive "$SPID"; do
  alive "$CPID" || { log "clients exited while the server was running"; break; }
  sleep 15
done
log "server exited after $(( $(date +%s) - T0 ))s"
teardown

# Learn steps are read from the checkpoints and the tensorboard by
# summarise.py; the server's log does not name them.
log "checkpoints: $(find "$OUT/ck" -name model.safetensors | sed "s|$OUT/ck/||" | sort | tr '\n' ' ')"
grep -h -E "Traceback|OutOfMemory|CUDA out of memory|Error" "$OUT/server.log" | head -5
log "peak MiB on cards 0 and 1: $(awk -F, 'NR>0 { if ($2>a) a=$2; if ($3>b) b=$3 } END { print a", "b }' "$OUT/gpu_mem.csv")"
log "E25_CELL_DONE $CELL"
