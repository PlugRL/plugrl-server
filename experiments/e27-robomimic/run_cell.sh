#!/usr/bin/env bash
# One cell of E27: a policy and an algorithm on robomimic's square task (low
# dimensional state, NutAssemblySquare), SEEDS concurrently. E24's run_cell.sh,
# with the environment changed:
#
#   bash run_cell.sh CELL POLICY POLICY_VARIANT ALGO ALGO_VARIANT ITERS [OUT_DIR]
#
# * The client is robomimic-v1 on `square-img`, from its own venv
#   (.venv-robomimic: robomimic v0.4.0, robosuite 1.4.1, mujoco 2.3.7), with
#   EGL rendering. Its agentview image is 84x84: no policy here reads images.
# * The client is not seeded. robomimic-v1 refuses a seed - it cannot reach
#   the randomness of the robosuite simulation underneath - so SEED varies
#   the server, which initialises the policy, and not the environment.
# * fpo-policy reads the four low-dimensional keys dppo-policy's square
#   config reads - 23 values - through `--policy.state-keys` (#61);
#   dppo-policy reads them through that config.
#
# BUFFER (default 4096) and BATCH (the algorithm variant's default when unset)
# come from the environment.
set -uo pipefail

CELL="$1"
POLICY="$2"
POLICY_VARIANT="$3"
ALGO="$4"
ALGO_VARIANT="$5"
ITERS="$6"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${7:-$HERE/results/$CELL}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9400}"
BUFFER="${BUFFER:-4096}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
CLIENT_DIR="${PLUGRL_ENV_CLIENT:-$SERVER_DIR/../plugrl-env-client}"
SERVER_PY="${PLUGRL_SERVER_PYTHON:-$SERVER_DIR/../plugrl-server/.venv/bin/python}"
CLIENT_PY="${PLUGRL_CLIENT_PYTHON:-$CLIENT_DIR/.venv-robomimic/bin/python}"
KEYS=(robot0_eef_pos robot0_eef_quat robot0_gripper_qpos object)
mkdir -p "$OUT"

POLICY_ARGS=()
REPLAN=1
if [ "$POLICY" = "fpo-policy" ]; then
  POLICY_ARGS=(--policy.obs-dim 23 --policy.action-dim 7 --policy.state-keys "${KEYS[@]}")
elif [ "$POLICY" = "dppo-policy" ]; then
  POLICY_ARGS=(--policy.env-type robomimic --policy.env-name square)
  REPLAN=4
fi

ALGO_ARGS=(--algo.buffer-size "$BUFFER")
[ -n "${BATCH:-}" ] && ALGO_ARGS+=(--algo.batch-size "$BATCH")
if [ "$ALGO" = "fpo" ]; then
  ALGO_ARGS+=(--algo.global-steps $((BUFFER * ITERS)))
else
  ALGO_ARGS+=(--algo.train-itrs "$ITERS")
fi

echo "cell:    $CELL = $POLICY/$POLICY_VARIANT x $ALGO/$ALGO_VARIANT x robomimic square"
echo "iters:   $ITERS x $BUFFER  batch: ${BATCH:-variant default}  replan: $REPLAN  seeds: $SEEDS"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
echo "client:  $CLIENT_PY"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
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

  (cd "$CLIENT_DIR" && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$CLIENT_PY" -m plugrl_env_client.cli robomimic-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((BUFFER * ITERS * 4)) \
      --env.name square-img --env.agentview-image-size 84 84 \
      --runner.replan-steps "$REPLAN" \
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
