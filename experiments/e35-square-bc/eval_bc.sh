#!/usr/bin/env bash
# Evaluate the behaviour-cloned fpo-policy on robomimic square, on guangzhao.
#
#   bash eval_bc.sh CHECKPOINT_DIR FLOW_STEPS EPISODES PORT OUT
#
# The `eval` algorithm (the flow's ODE, no sampling noise), one robomimic-v1
# client on `square-img` with the 84 x 84 agentview nothing reads, episodes of
# at most 400 steps that end on success, replanning every 4 steps. Success is
# counted by the client (a reward of 1 at any step); the server writes the
# episode statistics.
set -uo pipefail

CKPT="$1"
FLOW_STEPS="$2"
EPISODES="$3"
PORT="$4"
OUT="$5"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
HOME_DIR="$ROOT/.."
SERVER_PY="$HOME_DIR/plugrl-server/.venv/bin/python"
CLIENT_DIR="$HOME_DIR/plugrl-env-client"
CLIENT_PY="$CLIENT_DIR/.venv-robomimic/bin/python"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1

echo "checkpoint: $CKPT sha256 $(sha256sum "$CKPT/model.safetensors" | cut -c1-16)"
echo "flow steps: $FLOW_STEPS  episodes: $EPISODES"
date '+start %F %T'

(cd "$ROOT" && PYTHONPATH="$ROOT/src" "$SERVER_PY" -m plugrl_server.cli \
    fpo-policy default eval default \
    --port "$PORT" --seed 0 --policy.device cpu \
    --policy.obs-dim 23 --policy.action-dim 7 --policy.action-horizon 4 \
    --policy.flow-steps "$FLOW_STEPS" --policy.hidden-dims 1024 1024 1024 \
    --policy.state-keys robot0_eef_pos robot0_eef_quat robot0_gripper_qpos object \
    --algo.policy-checkpoint-path "$CKPT" --algo.num-episodes "$EPISODES" \
    --no-show-progress-bar --no-show-metric-table \
    --checkpoint-base-dir "$OUT/ck" --exp-name "bc-flow$FLOW_STEPS" --overwrite \
    > "$OUT/server.log" 2>&1 &)

for _ in $(seq 300); do
  grep -q "is listening on" "$OUT/server.log" 2>/dev/null && break
  sleep 1
done

(cd "$CLIENT_DIR" && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$CLIENT_PY" -m plugrl_env_client.cli robomimic-v1 \
    --server-port "$PORT" --server-host 127.0.0.1 \
    --num-envs 1 --num-episodes "$EPISODES" \
    --env.name square-img --env.agentview-image-size 84 84 \
    --runner.replan-steps 4 --exp-name "bc-flow$FLOW_STEPS-$PORT" \
    > "$OUT/client.log" 2>&1)
echo "client exited rc=$?"
pkill -f -- "--port $PORT " 2>/dev/null
SUMMARY=$(ls -dt "$CLIENT_DIR"/runs/bc-flow"$FLOW_STEPS-$PORT"*/rollout/proc_000/summary.json 2>/dev/null | head -1)
cp "$SUMMARY" "$OUT/client-summary.json"
"$SERVER_PY" -c "
import json, sys
s = json.load(open(sys.argv[1]))
print('episodes', s['completed_episodes'], 'window', s['metric_window'],
      'success rate', s['mean_success_rate'], 'mean return', s['mean_return'])
" "$OUT/client-summary.json"
date '+end   %F %T'
echo BC_EVAL_DONE
