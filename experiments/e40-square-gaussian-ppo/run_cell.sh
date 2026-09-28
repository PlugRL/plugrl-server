#!/usr/bin/env bash
# E40's cell: DPPO's Gaussian MLP under PPO on robomimic square, in DPPO's
# own Gaussian fine-tuning setting, SEEDS concurrently. E34's run_cell.sh with
# the server changed:
#
# * The server is `dppo-gaussian-policy default ppo dppo-square`: DPPO's
#   Gaussian MLP started from its released checkpoint (CKPT, its `model`
#   weights), and every value of DPPO's ft_ppo_gaussian_mlp.yaml for square.
#   The variant's own buffer and batch are used; only the iteration count and
#   the save interval are passed.
# * Each seed runs NPROC client processes (default 2) against its server, so
#   an iteration of 80,000 environment steps is collected in parallel.
# * Episodes run their full 400 steps and do not end on success
#   (`--env.no-terminate-on-success`), as DPPO counts them: success is any
#   step's reward reaching 1.
#
#   bash run_cell.sh CELL ITERS [OUT_DIR]
set -uo pipefail

CELL="$1"
ITERS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/results/$CELL}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9750}"
NPROC="${NPROC:-2}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
CLIENT_DIR="${PLUGRL_ENV_CLIENT:-$SERVER_DIR/../plugrl-env-client}"
SERVER_PY="${PLUGRL_SERVER_PYTHON:-$SERVER_DIR/../plugrl-server/.venv/bin/python}"
CLIENT_PY="${PLUGRL_CLIENT_PYTHON:-$CLIENT_DIR/.venv-robomimic/bin/python}"
CKPT="${CKPT:?set CKPT to the released DPPO square Gaussian checkpoint}"
mkdir -p "$OUT"

echo "cell:    $CELL = dppo-gaussian-policy x ppo/dppo-square x robomimic square"
echo "iters:   $ITERS  clients per seed: $NPROC  seeds: $SEEDS"
echo "ckpt:    $CKPT sha256 $(sha256sum "$CKPT" | cut -c1-16)"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
echo "client:  $CLIENT_PY"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
      dppo-gaussian-policy default ppo dppo-square \
      --port "$port" --seed "$seed" --policy.device cpu \
      --policy.checkpoint-path "$CKPT" \
      --algo.train-itrs "$ITERS" --algo.save-interval 10 \
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
      --num-envs 1 --num-procs "$NPROC" --num-episodes 100000000 \
      --env.name square-img --env.agentview-image-size 84 84 \
      --env.no-terminate-on-success \
      --runner.replan-steps 4 \
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
