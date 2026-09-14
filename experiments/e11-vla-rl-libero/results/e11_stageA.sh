#!/usr/bin/env bash
# E11 Stage A harness, per experiments/e11-vla-rl-libero/PROTOCOL.md.
#
#   bash e11_stageA.sh CELL SUITE EPISODES_PER_TASK PORT [BUDGET_SECONDS]
#
# One server runs the eval algorithm with no episode limit. Ten env client
# processes, one per task via --runner.pass-proc-id, each stop after their own
# EPISODES_PER_TASK. The server's own episode limit is a total across all
# processes, so using it would let fast tasks take episodes from slow ones.
#
# Rows go to $E11_RES, default $E/results. A harness check that is not one of
# the protocol's cells sets E11_RES elsewhere, so its rows never mix with them.
#
# Process hygiene. An env client retries forever while its server is away, and
# LIBERO's multiprocessing workers run as "python -c from multiprocessing.spawn
# ...", so killing by the CLI's name misses them. Once, workers left over from a
# killed cell joined the next cell's server the moment it listened and ran
# episodes into its metrics. So: refuse to start while any such process is
# alive, give the server and the clients each their own session, take down the
# whole process group, and check that nothing survived.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e11
RES=${E11_RES:-$E/results}
mkdir -p "$RES"

CELL="$1"
SUITE="$2"
NEP="$3"
PORT="$4"
BUDGET_S="${5:-28800}"
NPROC=10

SPY=$R/venv/bin/python
CPY=$R/venv-libero/bin/python
UV=$HOME/.local/bin/uv
OUT=$E/runs-$CELL

CLIENT_PATTERN="[v]env-libero/bin/python"
SERVER_PATTERN="[p]lugrl_server.cli pi0-policy"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# kill -0 succeeds on a zombie, and a child that has exited stays a zombie
# until it is waited for, so liveness has to read the process state.
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

# Both packages run from source trees with no .git, so pip freeze records only
# a path. Hash every source file, carriage returns stripped so each hash can be
# compared with the git blob, to tie a cell to the code that produced it.
# find_spec locates the package without importing it.
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

log "cell=$CELL suite=$SUITE episodes_per_task=$NEP procs=$NPROC port=$PORT budget=${BUDGET_S}s res=$RES"

if ! report_strays "refusing to start, env client processes" "$CLIENT_PATTERN" \
  || ! report_strays "refusing to start, pi0 server processes" "$SERVER_PATTERN"; then
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

# The exact environment of both processes, recorded before anything runs.
"$UV" pip freeze --python "$SPY" > "$RES/environment-server.txt" 2>/dev/null
"$UV" pip freeze --python "$CPY" > "$RES/environment-client.txt" 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$RES/environment-gpu.txt" 2>/dev/null
{
  source_manifest "$SPY" plugrl_server
  source_manifest "$CPY" plugrl_env_client
} > "$RES/environment-source-$CELL.txt"
log "source manifest: $(wc -l < "$RES/environment-source-$CELL.txt") files, sha256 $(sha256sum "$RES/environment-source-$CELL.txt" | cut -c1-16)"

free_port "$PORT"

# exec setsid in a background subshell: the subshell is not a group leader, so
# setsid does not fork, and $! is the server's own pid and process group.
(
  cd "$R/plugrl-server" && exec setsid env \
    OPENPI_DATA_HOME="$R/.cache/openpi" XDG_CACHE_HOME="$R/.cache" TMPDIR="$R/.tmp" \
    HF_HOME="$R/.cache/hf" TORCHINDUCTOR_CACHE_DIR="$R/.cache/inductor" CUDA_VISIBLE_DEVICES=0 \
    "$SPY" -m plugrl_server.cli pi0-policy default eval default \
      --policy.name pi05_libero \
      --policy.checkpoint-path "$R/ckpt/pi05_libero" \
      --policy.device cuda \
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

# Pi0Policy takes about 70 s to construct; a cold import off Lustre adds more.
for _ in $(seq 600); do
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
      --num-envs 1 --num-procs "$NPROC" --num-episodes "$NEP" \
      --env.task-suite-name "$SUITE" \
      --env.no-randomize-initial-state \
      --runner.pass-proc-id \
      --runner.replan-steps 5 \
      --runner.seed 7 \
      --recorder.no-thread0-only \
      --exp-name "$CELL"
) > "$OUT/client.log" 2>&1 &
CPID=$!

# A client whose server has died retries forever, so the clients are stopped
# as soon as the server is gone rather than at the end of the budget.
SERVER_DIED=0
while alive "$CPID"; do
  if ! alive "$SPID"; then
    log "server exited while clients were running"
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
log "clients exited with $CEXIT after ${WALL}s"

# The eval server has no episode limit, so it has to be taken down.
teardown

"$SPY" "$E/e11_stageA_post.py" --cell "$CELL" --suite "$SUITE" --nep "$NEP" --nproc "$NPROC" \
  --out "$OUT" --res "$RES" --client-exit "$CEXIT" --wall "$WALL"
log "STAGEA_CELL_DONE $CELL"
