#!/usr/bin/env bash
# E46, after the registered run (not pre-registered): is v2's shorter infer
# wait in Part B the arm, or running three at once? One run at a time:
# v1, v2, v2, v1, each STEPS steps of E43's cell, seed 0.
#   bash explore_alone.sh STEPS OUT_DIR
set -uo pipefail
STEPS="$1"
OUT="$2"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
E46="$BASE/e46"
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
n=0
for arm in v1 v2 v2 v1; do
  n=$((n + 1))
  name="$arm-$n"
  flag=""
  [ "$arm" = v1 ] && flag="--no-reuse-feedback-obs"
  port=$((9973 + n))
  (cd "$OUT" && exec env PYTHONPATH="$E46/server-src:$E46/protocol-src" "$BASE/plugrl-server/.venv/bin/python" -m plugrl_server.cli \
      fpo-policy default fpo default --port "$port" --seed 0 --policy.device cpu \
      --algo.global-steps "$STEPS" --algo.buffer-size 4096 --algo.save-interval 5 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-$name" --overwrite \
      > "$OUT/server-$name.log" 2>&1) &
  pid=$!
  until grep -q "is listening on" "$OUT/server-$name.log" 2>/dev/null; do sleep 1; done
  (cd "$OUT" && env PYTHONPATH="$E46/client-src:$E46/protocol-src" "$BASE/plugrl-env-client/.venv/bin/python" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed 0 --exp-name "e46-$name" $flag \
      > "$OUT/client-$name.log" 2>&1)
  wait "$pid"
  python3 -c "import json; d=json.load(open('$OUT/runs/e46-$name/rollout/proc_000/summary.json')); t=d.get('timing',d); print('$name', 'infer_wait per call %.3f ms' % (t['infer_wait_s']/t['infer_calls']*1e3), 'collect %.1f s' % t['collect_time_s'])"
done
echo EXPLORE_DONE
