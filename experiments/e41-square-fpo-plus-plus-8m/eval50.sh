#!/usr/bin/env bash
# Fifty evaluation episodes each for E41's final checkpoints: E39's eval50.sh
# with E41's results in place of E39's. The clone is not run again; E39's
# fifty episodes of it, 0.50, used this same script. A reported figure of
# E41's protocol.
set -uo pipefail
P=~/zuogou/plugrl
R=$P/e41-reg/experiments/e41-square-fpo-plus-plus-8m/results/fpopp-square-8m/fpo/fpo-policy
O=$P/eval-e41
SERVER_PY=$P/plugrl-server/.venv/bin/python
CLIENT_DIR=$P/plugrl-env-client
CLIENT_PY=$CLIENT_DIR/.venv-robomimic/bin/python
ROOT=$P/e41-reg
mkdir -p "$O"
export OMP_NUM_THREADS=1

last() { ls "$1" | grep -E '^[0-9]+$' | sort -n | tail -1; }

run() {  # NAME CKPT PORT VALUE_DIMS...
  local name="$1" ckpt="$2" port="$3"; shift 3
  local out="$O/$name"
  mkdir -p "$out"
  (cd "$ROOT" && PYTHONPATH="$ROOT/src" "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default eval default \
      --port "$port" --seed 0 --policy.device cpu \
      --policy.obs-dim 23 --policy.action-dim 7 --policy.action-horizon 4 \
      --policy.flow-steps 10 --policy.hidden-dims 1024 1024 1024 \
      --policy.value-hidden-dims "$@" \
      --policy.state-keys robot0_eef_pos robot0_eef_quat robot0_gripper_qpos object \
      --algo.policy-checkpoint-path "$ckpt" --algo.num-episodes 50 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$out/ck" --exp-name "$name" --overwrite \
      > "$out/server.log" 2>&1 &)
  for _ in $(seq 300); do
    grep -q "is listening on" "$out/server.log" 2>/dev/null && break
    sleep 1
  done
  (cd "$CLIENT_DIR" && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$CLIENT_PY" -m plugrl_env_client.cli robomimic-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes 50 \
      --env.name square-img --env.agentview-image-size 84 84 \
      --runner.replan-steps 4 --exp-name "e41eval-$name-$port" \
      > "$out/client.log" 2>&1)
  echo "$name client rc=$?"
  pkill -f "[-]-port $port " 2>/dev/null
  local summary
  summary=$(ls -dt "$CLIENT_DIR"/runs/e41eval-"$name-$port"*/rollout/proc_000/summary.json 2>/dev/null | head -1)
  cp "$summary" "$out/client-summary.json"
  "$SERVER_PY" -c "
import json, sys
s = json.load(open(sys.argv[1]))
print(sys.argv[2], 'episodes', s['completed_episodes'], 'success rate', s['mean_success_rate'])
" "$out/client-summary.json" "$name"
}

for s in 0 1 2; do
  d=$R/fpopp-square-8m-seed$s
  run "fpopp8m-s$s" "$d/$(last "$d")" $((9920 + s)) 512 256 &
  sleep 5
done
wait
echo EVAL_E41_DONE
