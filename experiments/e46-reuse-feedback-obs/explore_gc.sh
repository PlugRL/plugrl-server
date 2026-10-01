#!/usr/bin/env bash
# E46, after the registered run (not pre-registered): is v1's longer infer
# wait Python's garbage collector? v1 runs with gc disabled in the server
# only, then in the client only, against v1 and v2 with it on.
#   bash explore_gc.sh STEPS OUT_DIR
set -uo pipefail
STEPS="$1"
OUT="$2"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
E46="$BASE/e46"
NOGC="$E46/nogc"
mkdir -p "$NOGC" "$OUT"
printf 'import gc\ngc.disable()\n' > "$NOGC/sitecustomize.py"
export OMP_NUM_THREADS=1
n=0
for spec in v1:on:on v1:off:on v1:on:off v2:on:on; do
  IFS=: read -r arm sgc cgc <<< "$spec"
  n=$((n + 1))
  name="$arm-server_gc_$sgc-client_gc_$cgc"
  flag=""
  [ "$arm" = v1 ] && flag="--no-reuse-feedback-obs"
  spath="$E46/server-src:$E46/protocol-src"
  cpath="$E46/client-src:$E46/protocol-src"
  [ "$sgc" = off ] && spath="$NOGC:$spath"
  [ "$cgc" = off ] && cpath="$NOGC:$cpath"
  port=$((9980 + n))
  (cd "$OUT" && exec env PYTHONPATH="$spath" "$BASE/plugrl-server/.venv/bin/python" -m plugrl_server.cli \
      fpo-policy default fpo default --port "$port" --seed 0 --policy.device cpu \
      --algo.global-steps "$STEPS" --algo.buffer-size 4096 --algo.save-interval 5 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-$n" --overwrite \
      > "$OUT/server-$n.log" 2>&1) &
  pid=$!
  until grep -q "is listening on" "$OUT/server-$n.log" 2>/dev/null; do sleep 1; done
  (cd "$OUT" && env PYTHONPATH="$cpath" "$BASE/plugrl-env-client/.venv/bin/python" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed 0 --exp-name "e46-gc-$n" $flag \
      > "$OUT/client-$n.log" 2>&1)
  wait "$pid"
  python3 -c "import json; d=json.load(open('$OUT/runs/e46-gc-$n/rollout/proc_000/summary.json')); t=d.get('timing',d); print('$name', 'infer_wait per call %.3f ms' % (t['infer_wait_s']/t['infer_calls']*1e3))"
done
echo EXPLORE_GC_DONE
