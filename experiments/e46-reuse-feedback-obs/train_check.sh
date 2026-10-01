#!/usr/bin/env bash
# E46 Part B, on guangzhao: does reusing observations change what is trained?
#
# E43's cell (fpo-policy · FPO · HalfCheetah-v5, E16's command lines), on one
# machine, seed 0, three runs at once:
#
#   v1a  the env client sends every observation (--no-reuse-feedback-obs)
#   v1b  the same again: the control for determinism
#   v2   the env client reuses what the server holds (the default)
#
# The server offers reuse-feedback-obs in all three, so the runs differ only
# in the client. Code: plugrl-server#109, plugrl-env-client#16 and
# plugrl-protocol#11, laid over guangzhao's existing venvs with PYTHONPATH.
#
#   bash train_check.sh STEPS OUT_DIR
set -uo pipefail

STEPS="$1"
OUT="$2"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac
BASE="$HOME/zuogou/plugrl"
E46="$BASE/e46"
SERVER_PY="$BASE/plugrl-server/.venv/bin/python"
CLIENT_PY="$BASE/plugrl-env-client/.venv/bin/python"
BUFFER=4096
SEED=0
export OMP_NUM_THREADS=1
mkdir -p "$OUT"
date '+start %F %T'

run_arm() {
  local arm="$1" port="$2" flag="$3"
  (cd "$OUT" && exec env PYTHONPATH="$E46/server-src:$E46/protocol-src" "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$SEED" --policy.device cpu \
      --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 5 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-$arm" --overwrite \
      > "$OUT/server-$arm.log" 2>&1) &
  local server_pid=$!
  local waited=0
  until grep -q "is listening on" "$OUT/server-$arm.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 180 ]; then
      echo "$arm: server never listened" >&2
      return 1
    fi
  done
  echo "READY $arm port $port at $(date '+%T')"
  (cd "$OUT" && env PYTHONPATH="$E46/client-src:$E46/protocol-src" "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed "$SEED" \
      --exp-name "e46-$arm" $flag \
      > "$OUT/client-$arm.log" 2>&1)
  echo "$arm client rc=$? at $(date '+%T')"
  wait "$server_pid"
  echo "$arm server rc=$? at $(date '+%T')"
}

run_arm v1a 9970 --no-reuse-feedback-obs &
A=$!
sleep 5
run_arm v1b 9971 --no-reuse-feedback-obs &
B=$!
sleep 5
run_arm v2 9972 "" &
C=$!
wait "$A" "$B" "$C"
date '+end   %F %T'
echo TRAIN_CHECK_DONE
