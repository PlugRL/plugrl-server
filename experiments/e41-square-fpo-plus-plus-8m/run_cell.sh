#!/usr/bin/env bash
# E41's cell: E39's three seeds resumed from their final checkpoints and run
# on to FPO++'s own 8M steps, SEEDS concurrently. E39's run_cell.sh with the
# start changed:
#
#   bash run_cell.sh CELL ITERS [OUT_DIR]
#
# * Each seed restores its own E39 run with `--algo.restore all`: weights,
#   both AdamW groups' state, the step count and the iteration count, so the
#   run continues at iteration 101 and stops when the step count reaches
#   ITERS x 12,000.
# * Every other setting is E39's, unchanged.
set -uo pipefail

CELL="$1"
ITERS="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/results/$CELL}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9700}"
NPROC="${NPROC:-3}"
SERVER_DIR="$(cd "$HERE/../.." && pwd)"
CLIENT_DIR="${PLUGRL_ENV_CLIENT:-$SERVER_DIR/../plugrl-env-client}"
SERVER_PY="${PLUGRL_SERVER_PYTHON:-$SERVER_DIR/../plugrl-server/.venv/bin/python}"
CLIENT_PY="${PLUGRL_CLIENT_PYTHON:-$CLIENT_DIR/.venv-robomimic/bin/python}"
E39="${E39:?set E39 to the run directory of E39, .../fpopp-square/fpo/fpo-policy}"
KEYS=(robot0_eef_pos robot0_eef_quat robot0_gripper_qpos object)
mkdir -p "$OUT"

POLICY_ARGS=(--policy.obs-dim 23 --policy.action-dim 7 --policy.action-horizon 4
  --policy.hidden-dims 1024 1024 1024 --policy.state-keys "${KEYS[@]}"
  --policy.freeze-obs-stats --policy.flow-steps 10
  --policy.value-hidden-dims 512 256)
ALGO_ARGS=(--algo.output-mode u --algo.no-discretize-t-for-training
  --algo.cfm-loss-steps 4 --algo.cfm-loss-dims 7 --algo.cfm-loss-sum-over-steps
  --algo.cfm-loss-huber-delta 1.0
  --algo.ratio-per-sample --algo.n-critic-warmup-itrs 1
  --algo.clipping-epsilon 0.01
  --algo.learning-rate 1e-5 --algo.critic-learning-rate 1e-4
  --algo.adam-eps 1e-5 --algo.weight-decay 1e-6 --algo.actor-adam-beta2 0.99
  --algo.max-grad-norm 25.0 --algo.normalize-advantage-per-minibatch
  --algo.no-fpo-playground-trick --algo.reward-scaling 1.0
  --algo.treat-truncated-as-done --algo.value-loss-coeff 0.5
  --algo.num-updates-per-batch 10 --algo.n-samples-per-action 8
  --algo.discounting 0.995 --algo.gae-lambda 0.99
  --algo.buffer-size 12000 --algo.batch-size 1500
  --algo.global-steps $((12000 * ITERS)) --algo.save-interval 10)

last() { ls "$1" | grep -E '^[0-9]+$' | sort -n | tail -1; }

echo "cell:    $CELL = E39's fpopp-square resumed to $ITERS iterations"
echo "seeds:   $SEEDS  clients per seed: $NPROC"
echo "from:    $E39"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
echo "client:  $CLIENT_PY"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  local run="$E39/fpopp-square-seed$seed"
  local ckpt="$run/$(last "$run")"
  echo "seed $seed resumes $ckpt"
  (cd "$SERVER_DIR" && PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      "${POLICY_ARGS[@]}" \
      --algo.policy-checkpoint-path "$ckpt" --algo.restore all \
      "${ALGO_ARGS[@]}" \
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
