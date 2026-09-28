#!/usr/bin/env bash
# One E37 cell: fpo-policy on robomimic square from E35's behaviour-cloned
# checkpoint, SEEDS concurrently. E34's run_cell.sh with the policy and the
# algorithm changed:
#
#   bash run_cell.sh CELL ALGO ITERS [OUT_DIR]      ALGO: fpo | dppo
#
# * The policy is fpo-policy as E35 cloned it - chunks of 4, three hidden
#   layers of 1024, the four low-dimensional keys - started from BC (the
#   cloned checkpoint) with `--algo.restore except-critic`, so the value head
#   starts from the run's own initialisation, and the observation statistics
#   stay the demonstrations' (`--policy.freeze-obs-stats`, #82): the first
#   pilot showed them drifting under a critic-only iteration and the cloned
#   policy's success falling with them.
# * fpo: FPO++'s square fine-tuning as far as this FPO has it (PROTOCOL.md):
#   the chunk loss over the 4 executed steps and 7 dimensions, velocity error,
#   uniform times, one ratio per sample, one critic-only iteration, clip 0.01,
#   learning rate 1e-5, 10 epochs of 8 minibatches, 8 samples per action,
#   gamma 0.995, lambda 0.99, 12,000 chunks (48,000 steps) per iteration,
#   10 flow steps.
# * dppo: `dppo square` (#77) with E33's two changes for a flow policy -
#   noise level 1.0 for sampling and the log-probability, 20 flow steps - and
#   minibatches of 500 entries, so each is DPPO's 10,000 (chunk, step)
#   samples.
# * Clients as E34's: NPROC per seed, episodes of 400 steps that do not end
#   on success, replanning every 4.
set -uo pipefail

CELL="$1"
ALGO="$2"
ITERS="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${4:-$HERE/results/$CELL}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac

SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-9600}"
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
  --policy.freeze-obs-stats)
START=(--algo.policy-checkpoint-path "$BC" --algo.restore except-critic)
if [ "$ALGO" = fpo ]; then
  VARIANT=default
  POLICY_ARGS+=(--policy.flow-steps 10)
  ALGO_ARGS=(--algo.output-mode u --algo.no-discretize-t-for-training
    --algo.cfm-loss-steps 4 --algo.cfm-loss-dims 7 --algo.cfm-loss-sum-over-steps
    --algo.ratio-per-sample --algo.n-critic-warmup-itrs 1
    --algo.clipping-epsilon 0.01 --algo.learning-rate 1e-5
    --algo.num-updates-per-batch 10 --algo.n-samples-per-action 8
    --algo.discounting 0.995 --algo.gae-lambda 0.99
    --algo.buffer-size 12000 --algo.batch-size 1500
    --algo.global-steps $((12000 * ITERS)) --algo.save-interval 10)
elif [ "$ALGO" = dppo ]; then
  VARIANT=square
  POLICY_ARGS+=(--policy.flow-steps 20)
  ALGO_ARGS=(--algo.sampling-noise-level 1.0 --algo.logprob-noise-level 1.0
    --algo.batch-size 500 --algo.train-itrs "$ITERS" --algo.save-interval 10)
else
  echo "ALGO must be fpo or dppo, got $ALGO" >&2
  exit 2
fi

echo "cell:    $CELL = fpo-policy x $ALGO/$VARIANT x robomimic square"
echo "iters:   $ITERS  clients per seed: $NPROC  seeds: $SEEDS"
echo "start:   $BC sha256 $(sha256sum "$BC/model.safetensors" | cut -c1-16)"
echo "server:  $SERVER_PY (src $SERVER_DIR/src)"
echo "client:  $CLIENT_PY"
echo "out:     $OUT"
date '+start %F %T'

run_seed() {
  local seed="$1" port=$((PORT_BASE + $1))
  (cd "$SERVER_DIR" && PYTHONPATH="$SERVER_DIR/src" "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default "$ALGO" "$VARIANT" \
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
