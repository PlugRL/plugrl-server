#!/usr/bin/env bash
# E39's one cell: fpo-policy on robomimic square from E35's behaviour-cloned
# checkpoint, under FPO with FPO++'s square fine-tuning in full, SEEDS
# concurrently. E37's run_cell.sh with the FPO settings completed:
#
#   bash run_cell.sh CELL ITERS [OUT_DIR]
#
# * The policy is E37's: fpo-policy as E35 cloned it (chunks of 4, three
#   hidden layers of 1024, the four low-dimensional keys), restored with
#   `except-critic` and its observation statistics frozen (#82). The value
#   head is FPO++'s shape, two layers of 512 and 256.
# * FPO: E37's policy loss (the chunk loss over 4 executed steps and 7
#   dimensions, velocity error, uniform times, a ratio per sample, clip 0.01,
#   8 samples per action, 10 flow steps) with what E37 lacked of FPO++'s
#   (amazon-far/fpo-control `manipulation_experiments`): a Huber error of
#   delta 1; AdamW, the actor at 1e-5 with betas (0.9, 0.99), the critic at
#   1e-4, both eps 1e-5 and weight decay 1e-6; each clipped to a gradient
#   norm of 25; advantages normalised per minibatch and computed once per
#   iteration; raw rewards; a truncated episode treated as ended; the value
#   loss as 0.5 x squared error. As before: one critic-only iteration, 10
#   epochs of 8 minibatches, gamma 0.995, lambda 0.99, 12,000 chunks (48,000
#   steps) per iteration.
# * Clients: NPROC per seed, episodes of at most 400 steps that end on
#   success, as FPO++'s do, replanning every 4.
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
BC="${BC:?set BC to the cloned checkpoint directory}"
KEYS=(robot0_eef_pos robot0_eef_quat robot0_gripper_qpos object)
mkdir -p "$OUT"

POLICY_ARGS=(--policy.obs-dim 23 --policy.action-dim 7 --policy.action-horizon 4
  --policy.hidden-dims 1024 1024 1024 --policy.state-keys "${KEYS[@]}"
  --policy.freeze-obs-stats --policy.flow-steps 10
  --policy.value-hidden-dims 512 256)
START=(--algo.policy-checkpoint-path "$BC" --algo.restore except-critic)
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

echo "cell:    $CELL = fpo-policy x fpo (FPO++ square) x robomimic square"
echo "iters:   $ITERS  clients per seed: $NPROC  seeds: $SEEDS"
echo "start:   $BC sha256 $(sha256sum "$BC/model.safetensors" | cut -c1-16)"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
echo "client:  $CLIENT_PY"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      "${POLICY_ARGS[@]}" "${START[@]}" "${ALGO_ARGS[@]}" \
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
