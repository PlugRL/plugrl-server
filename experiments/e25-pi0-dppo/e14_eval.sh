#!/usr/bin/env bash
# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the
# output paths differ, so the measurement is E11's, unchanged.
#
#   bash e11_stageC_eval.sh CELL POLICY_LABEL TASK_ID EPISODES PORT [BUDGET_SECONDS] [CHECKPOINT_DIR]
#
# Evaluates one policy on one libero_10 task: the baseline when CHECKPOINT_DIR
# is empty, a fine-tuned checkpoint otherwise. One env client process runs
# EPISODES episodes with initial states taken in order, 0 to EPISODES-1, so the
# baseline and the fine-tuned policy face exactly the same initial states. This
# is also how openpi evaluates LIBERO. Ten processes on one task would each
# start again from initial state 0 and repeat the same few states.
#
# Process hygiene is the Stage A harness's.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
RES=${E14_RES:-$E/results}
mkdir -p "$RES"

CELL="$1"
POLICY_LABEL="$2"
TASK_ID="$3"
NEP="$4"
PORT="$5"
BUDGET_S="${6:-28800}"
CKPT_DIR="${7:-}"
SUITE=libero_10
NPROC=1

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
  for _ in $(seq 10); do
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

log "cell=$CELL policy=$POLICY_LABEL suite=$SUITE task_id=$TASK_ID episodes=$NEP procs=$NPROC port=$PORT budget=${BUDGET_S}s checkpoint=${CKPT_DIR:-none} res=$RES"

if [ -n "$CKPT_DIR" ] && [ ! -f "$CKPT_DIR/model.safetensors" ]; then
  log "no model.safetensors in $CKPT_DIR"
  exit 2
fi

if [ "${ALLOW_SIBLINGS:-0}" = 1 ]; then
  log "ALLOW_SIBLINGS=1: not checking for other clients or servers"
elif ! report_strays "refusing to start, env client processes" "$CLIENT_PATTERN" \
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
if [ -n "$CKPT_DIR" ]; then
  log "checkpoint sha256: $(sha256sum "$CKPT_DIR/model.safetensors" | cut -c1-16)"
fi

CKPT_ARGS=()
if [ -n "$CKPT_DIR" ]; then
  CKPT_ARGS=(--algo.policy-checkpoint-path "$CKPT_DIR")
fi

free_port "$PORT"

(
  cd "$R/plugrl-server" && exec setsid env \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" CUDA_VISIBLE_DEVICES=${EVAL_SRV_GPU:-0} \
    "$SPY" -m plugrl_server.cli pi0-policy default eval default \
      --policy.name pi05_libero \
      --policy.checkpoint-path "$R/ckpt/pi05_libero" \
      --policy.device cuda \
      "${CKPT_ARGS[@]}" \
      --port "$PORT" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT/ck" --exp-name "$CELL" --overwrite
) > "$OUT/server.log" 2>&1 &
SPID=$!

teardown() {
  kill_group "$SPID"
  [ -n "${CPID:-}" ] && kill_group "$CPID"
  free_port "$PORT"
  if report_strays "after teardown, env client processes" "$CLIENT_PATTERN" \
    && report_strays "after teardown, pi0 server processes" "$SERVER_PATTERN"; then
    log "teardown clean"
  else
    log "TEARDOWN_NOT_CLEAN"
  fi
}

# Cold, after hours out of Lustre's cache, the server has taken over 600 s to
# import and construct.
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
    CUDA_VISIBLE_DEVICES=${EVAL_CLI_GPU:-1} MUJOCO_EGL_DEVICE_ID=${EVAL_CLI_GPU:-1} \
    XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    timeout "$BUDGET_S" "$CPY" -m plugrl_env_client.cli libero-v1 \
      --server-host 127.0.0.1 --server-port "$PORT" \
      --num-envs 1 --num-procs "$NPROC" --num-episodes "$NEP" \
      --env.task-suite-name "$SUITE" \
      --env.task-id "$TASK_ID" \
      --env.no-randomize-initial-state \
      --runner.replan-steps 5 \
      --runner.seed 7 \
      --recorder.no-thread0-only \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

SERVER_DIED=0
while alive "$CPID"; do
  if ! alive "$SPID"; then
    log "server exited while the client was running"
    SERVER_DIED=1
    kill_group "$CPID"
    break
  fi
  sleep 5
done
wait "$CPID"
CEXIT=$?
[ "$SERVER_DIED" = 1 ] && [ "$CEXIT" = 0 ] && CEXIT=97
WALL=$(( $(date +%s) - T0 ))
log "client exited with $CEXIT after ${WALL}s"

teardown

"$SPY" "$R/e11/e11_stageC_eval_post.py" --cell "$CELL" --policy-label "$POLICY_LABEL" --task-id "$TASK_ID" \
  --nep "$NEP" --out "$OUT" --res "$RES" --client-exit "$CEXIT" --wall "$WALL"
log "STAGEC_EVAL_DONE $CELL"
