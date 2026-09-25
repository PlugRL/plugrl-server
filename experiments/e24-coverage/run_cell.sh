#!/usr/bin/env bash
# One cell of E24's coverage matrix: a policy, an algorithm and a MuJoCo task,
# run on SEEDS concurrently, each seed a server and an env client.
#
#   bash run_cell.sh CELL POLICY POLICY_VARIANT ALGO ALGO_VARIANT ENV ITERS [OUT_DIR]
#
#   CELL            a name for the directory and the checkpoints
#   POLICY          fpo-policy | dppo-policy
#   ALGO            fpo | dppo
#   ENV             a gymnasium MuJoCo task, e.g. Walker2d-v5
#   ITERS           iterations of BUFFER entries
#
# Settings are E17-E23's wherever they apply: buffer 4,096, save every 20
# iterations, server and client seeds matched. What differs by combination:
#
# * fpo-policy takes the task's dimensions on the command line; dppo-policy
#   reads them, and its D4RL normalisation, from the config its variant names.
# * FPO counts --algo.global-steps; DPPO derives it from --algo.train-itrs.
# * dppo-policy returns a chunk of 4 actions, and DPPO's own setting executes
#   the whole chunk before the next inference, so its client replans every 4
#   steps. fpo-policy's chunk is 1.
# * The client's episode budget is an upper bound large enough never to bind:
#   the server ends every run.
#
# BUFFER (default 4096) and BATCH (the algorithm variant's own default when
# unset) come from the environment, so run.sh can set them per cell.
set -uo pipefail

CELL="$1"
POLICY="$2"
POLICY_VARIANT="$3"
ALGO="$4"
ALGO_VARIANT="$5"
ENV_NAME="$6"
ITERS="$7"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${8:-$HERE/results/$CELL}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9300}"
BUFFER="${BUFFER:-4096}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
mkdir -p "$OUT"

case "$ENV_NAME" in
  HalfCheetah-v5) DIMS=(17 6) ;;
  Hopper-v5) DIMS=(11 3) ;;
  Walker2d-v5) DIMS=(17 6) ;;
  *) echo "error: no dimensions recorded for $ENV_NAME" >&2; exit 1 ;;
esac

POLICY_ARGS=()
REPLAN=1
if [ "$POLICY" = "fpo-policy" ]; then
  POLICY_ARGS=(--policy.obs-dim "${DIMS[0]}" --policy.action-dim "${DIMS[1]}")
elif [ "$POLICY" = "dppo-policy" ]; then
  REPLAN=4
fi

ALGO_ARGS=(--algo.buffer-size "$BUFFER")
if [ -n "${BATCH:-}" ]; then
  ALGO_ARGS+=(--algo.batch-size "$BATCH")
fi
if [ "$ALGO" = "fpo" ]; then
  ALGO_ARGS+=(--algo.global-steps $((BUFFER * ITERS)))
else
  ALGO_ARGS+=(--algo.train-itrs "$ITERS")
fi

find_client_dir() {
  local candidate
  for candidate in \
      "${PLUGRL_ENV_CLIENT:-}" \
      "$SERVER_DIR/../plugrl-env-client" \
      "$SERVER_DIR/../../../../plugrl-env-client"; do
    [ -n "$candidate" ] && [ -d "$candidate" ] && (cd "$candidate" && pwd) && return
  done
  return 1
}
CLIENT_DIR="$(find_client_dir)" || {
  echo "error: no plugrl-env-client found. Set PLUGRL_ENV_CLIENT." >&2
  exit 1
}

pick_python() {
  local repo="$1" override="${2:-}" candidate
  if [ -n "$override" ]; then echo "$override"; return; fi
  for candidate in "$repo/.venv/Scripts/python.exe" "$repo/.venv/bin/python"; do
    [ -x "$candidate" ] && { echo "$candidate"; return; }
  done
  command -v python3 || command -v python
}
SERVER_PY="$(pick_python "$SERVER_DIR" "${PLUGRL_SERVER_PYTHON:-}")"
CLIENT_PY="$(pick_python "$CLIENT_DIR" "${PLUGRL_CLIENT_PYTHON:-}")"

echo "cell:    $CELL = $POLICY/$POLICY_VARIANT x $ALGO/$ALGO_VARIANT x $ENV_NAME"
echo "iters:   $ITERS x $BUFFER  batch: ${BATCH:-variant default}  replan: $REPLAN  seeds: $SEEDS"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      "$POLICY" "$POLICY_VARIANT" "$ALGO" "$ALGO_VARIANT" \
      --port "$port" --seed "$seed" --policy.device cpu \
      "${POLICY_ARGS[@]}" "${ALGO_ARGS[@]}" \
      --algo.save-interval 20 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "$CELL-seed$seed" --overwrite \
      > "$OUT/server-seed$seed.log" 2>&1 &)

  local waited=0
  until grep -q "is listening on" "$OUT/server-seed$seed.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 300 ]; then
      echo "seed $seed: server never listened; see $OUT/server-seed$seed.log" >&2
      return 1
    fi
  done

  (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((BUFFER * ITERS * 4)) \
      --env.name "$ENV_NAME" \
      --runner.replan-steps "$REPLAN" --runner.seed "$seed" \
      > "$OUT/client-seed$seed.log" 2>&1)
  echo "seed $seed finished rc=$? at $(date '+%T')"
}

PIDS=()
for seed in $SEEDS; do
  run_seed "$seed" &
  PIDS+=($!)
  sleep 5
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done

date '+end   %F %T'
echo "failed seeds: $FAILED"
echo "CELL_DONE $CELL"
