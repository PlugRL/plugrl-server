#!/usr/bin/env bash
# E11 Stage C training harness, per experiments/e11-vla-rl-libero/PROTOCOL.md.
#
#   bash e11_stageC_train.sh CELL TASK_ID ITERATIONS PORT [BUDGET_SECONDS] [BUFFER_SIZE]
#
# One FPO server fine-tunes pi05_libero with the protocol's starting values.
# Ten env client processes all run the chosen libero_10 task, with randomised
# initial states. The server trains for exactly ITERATIONS learn steps:
# global_steps is ITERATIONS * BUFFER_SIZE, a learn fires each time the buffer
# fills, and the server stops, saving a checkpoint, once global_steps is
# reached. The clients have no episode limit of their own and end when the
# server stops.
#
# BUFFER_SIZE defaults to the protocol's 4096. A harness check passes a small
# one, sets E11_RES elsewhere, and is never one of the protocol's runs.
#
# E11_FPO_BATCH_SIZE and E11_FPO_N_SAMPLES default to the protocol's starting
# values, 32 and 4. PROTOCOL.md allows at most two adjustments, each recorded in
# AMENDMENT.md before the run that uses it; an adjustment reaches the server
# only through these two variables, and the log line prints both.
#
# Process hygiene is the Stage A harness's: refuse to start while anything is
# left over, give the server and the clients their own sessions, take down
# whole process groups, and check that nothing survived.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e11
RES=${E11_RES:-$E/results}
mkdir -p "$RES"

CELL="$1"
TASK_ID="$2"
ITERATIONS="$3"
PORT="$4"
BUDGET_S="${5:-72000}"
BUFFER_SIZE="${6:-4096}"
BATCH_SIZE="${E11_FPO_BATCH_SIZE:-32}"
N_SAMPLES="${E11_FPO_N_SAMPLES:-4}"
SUITE=libero_10
NPROC=10
GLOBAL_STEPS=$(( ITERATIONS * BUFFER_SIZE ))

SPY=$R/venv/bin/python
CPY=$R/venv-libero/bin/python
UV=$HOME/.local/bin/uv
OUT=$E/runs-$CELL

CLIENT_PATTERN="[v]env-libero/bin/python"
SERVER_PATTERN="[p]lugrl_server.cli pi0-policy"

log() { echo "[$(date +%H:%M:%S)] $*"; }

alive() {
  local s
  s=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')
  [ -n "$s" ] && [ "${s#Z}" = "$s" ]
}

free_port() {
  local p="$1"
  for pid in $(ss -tlnp 2>/dev/null | awk -v pat=":$p" '$4 ~ pat {print $NF}' | grep -oP 'pid=\K[0-9]+' | sort -u); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 1
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
  pids=$(pgrep -d, -f "$pattern")
  if [ -n "$pids" ]; then
    log "$what: $(echo "$pids" | tr ',' '\n' | wc -l) process(es) alive"
    ps -o pid,ppid,pgid,etimes,args -p "$pids" | cut -c1-180
    return 1
  fi
  return 0
}

source_manifest() {
  local py="$1" pkg="$2" dir
  dir=$("$py" -c "import importlib.util as u; print(u.find_spec('$pkg').submodule_search_locations[0])" 2>/dev/null | tail -1)
  if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    echo "unavailable  $pkg"
    return
  fi
  (cd "$dir" && find . -name '*.py' -not -path '*/__pycache__/*' | sort | while read -r f; do
    printf '%s  %s/%s\n' "$(tr -d '\r' < "$f" | sha256sum | cut -d' ' -f1)" "$pkg" "${f#./}"
  done)
}

log "cell=$CELL suite=$SUITE task_id=$TASK_ID iterations=$ITERATIONS buffer=$BUFFER_SIZE batch_size=$BATCH_SIZE n_samples=$N_SAMPLES global_steps=$GLOBAL_STEPS procs=$NPROC port=$PORT budget=${BUDGET_S}s res=$RES"

if ! report_strays "refusing to start, env client processes" "$CLIENT_PATTERN" \
  || ! report_strays "refusing to start, pi0 server processes" "$SERVER_PATTERN"; then
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

"$UV" pip freeze --python "$SPY" > "$RES/environment-server.txt" 2>/dev/null
"$UV" pip freeze --python "$CPY" > "$RES/environment-client.txt" 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$RES/environment-gpu.txt" 2>/dev/null
{
  source_manifest "$SPY" plugrl_server
  source_manifest "$CPY" plugrl_env_client
} > "$RES/environment-source-$CELL.txt"
log "source manifest: $(wc -l < "$RES/environment-source-$CELL.txt") files, sha256 $(sha256sum "$RES/environment-source-$CELL.txt" | cut -c1-16)"

free_port "$PORT"

(
  cd "$R/plugrl-server" && exec setsid env \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" CUDA_VISIBLE_DEVICES=0 \
    "$SPY" -m plugrl_server.cli pi0-policy default fpo default \
      --policy.name pi05_libero \
      --policy.checkpoint-path "$R/ckpt/pi05_libero" \
      --policy.device cuda \
      --algo.learning-rate 1e-5 \
      --algo.batch-size "$BATCH_SIZE" \
      --algo.num-updates-per-batch 4 \
      --algo.n-samples-per-action "$N_SAMPLES" \
      --algo.buffer-size "$BUFFER_SIZE" \
      --algo.clipping-epsilon 0.05 \
      --algo.global-steps "$GLOBAL_STEPS" \
      --algo.save-interval 5 \
      --port "$PORT" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT/ck" --exp-name "$CELL" --overwrite
) > "$OUT/server.log" 2>&1 &
SPID=$!

# Peak GPU memory is a pre-registered column, and nothing in the server
# records it, so it is sampled from outside.
(
  exec setsid bash -c 'while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)"; sleep 15; done'
) > "$OUT/gpu0_mem.csv" 2>/dev/null &
MPID=$!

teardown() {
  kill_group "$SPID"
  [ -n "${CPID:-}" ] && kill_group "$CPID"
  kill_group "$MPID"
  free_port "$PORT"
  if report_strays "after teardown, env client processes" "$CLIENT_PATTERN" \
    && report_strays "after teardown, pi0 server processes" "$SERVER_PATTERN"; then
    log "teardown clean"
  else
    log "TEARDOWN_NOT_CLEAN"
  fi
}

# Pi0Policy constructs in about 70 s warm. Cold, after hours with the Python
# environment out of Lustre's cache, importing it has taken over 600 s.
for _ in $(seq 1500); do
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
    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
    XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    timeout "$BUDGET_S" "$CPY" -m plugrl_env_client.cli libero-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" \
      --num-envs 1 --num-procs "$NPROC" --num-episodes 1000000000 \
      --env.task-suite-name "$SUITE" \
      --env.task-id "$TASK_ID" \
      --env.randomize-initial-state \
      --runner.replan-steps 5 \
      --runner.seed 7 \
      --recorder.no-thread0-only \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

# The server ends a training run. Clients that die first leave a server
# waiting forever for feedback, so either side ending takes down the other.
SERVER_DIED=0
CLIENTS_DIED_FIRST=0
while alive "$CPID"; do
  if ! alive "$SPID"; then
    log "server exited"
    SERVER_DIED=1
    # A server that stops normally closes every connection itself; give the
    # clients a moment to write their final summaries before the group goes.
    for _ in $(seq 60); do alive "$CPID" || break; sleep 1; done
    kill_group "$CPID"
    break
  fi
  sleep 10
done
wait "$CPID"
CEXIT=$?
# The loop can see the clients gone before it sees the server gone: a server
# that stops normally closes every connection, and the clients follow within
# a second. Only a server still running now means the clients went first.
if [ "$SERVER_DIED" = 0 ]; then
  if alive "$SPID"; then
    CLIENTS_DIED_FIRST=1
    log "clients exited before the server"
  else
    SERVER_DIED=1
  fi
fi
WALL=$(( $(date +%s) - T0 ))
log "clients exited with $CEXIT after ${WALL}s (server_died=$SERVER_DIED clients_died_first=$CLIENTS_DIED_FIRST)"

teardown

log "checkpoints:"
find "$OUT/ck" -name model.safetensors -printf '%TY-%Tm-%Td %TH:%TM  %s  %p\n' 2>/dev/null | sort
log "STAGEC_TRAIN_DONE $CELL"
