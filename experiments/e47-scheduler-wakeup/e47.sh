#!/usr/bin/env bash
# E47 on guangzhao: the server's scheduler, polling (main) against waking on
# work (fix), with a reference arm that never sleeps (sleep0: main with
# SCHEDULER_SLEEP_INTERVAL = 0.0). All use guangzhao's E43 venvs and env
# client (931ab56); only the server's source differs, laid over with
# PYTHONPATH. Training compares main and fix only.
#
#   bash e47.sh rtt   REPS EXCHANGES OUT   # scripted v1 client, per-infer round trip
#   bash e47.sh idle  REPS SECONDS   OUT   # a server with no client: CPU it burns
#   bash e47.sh train STEPS          OUT   # E43's cell, seed 0, each arm alone
set -uo pipefail
MODE="$1"
shift
BASE="$HOME/zuogou/plugrl"
E47="$BASE/e47"
PY="$BASE/plugrl-server/.venv/bin/python"
CLIENT_PY="$BASE/plugrl-env-client/.venv/bin/python"
export OMP_NUM_THREADS=1

absolute() { case "$1" in /*) echo "$1" ;; *) echo "$(pwd)/$1" ;; esac; }

start_server() {  # $1 arm, $2 port, $3 steps, $4 out (absolute), $5 name
  (cd "$4" && exec env PYTHONPATH="$E47/$1-src" "$PY" -m plugrl_server.cli \
      fpo-policy default fpo default --port "$2" --seed 0 --policy.device cpu \
      --algo.global-steps "$3" --algo.buffer-size 4096 --algo.save-interval 5 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$4" --exp-name "$5" --overwrite \
      > "$4/server-$5.log" 2>&1) &
  SERVER_PID=$!
  local waited=0
  until grep -q "is listening on" "$4/server-$5.log" 2>/dev/null; do
    sleep 0.2
    waited=$((waited + 1))
    if [ "$waited" -ge 600 ]; then
      echo "server $5 never listened" >&2
      exit 1
    fi
  done
}

case "$MODE" in
  rtt)
    REPS="$1"; N="$2"; OUT="$(absolute "$3")"; mkdir -p "$OUT"
    printf 'arm\trep\tms_per_round_trip\n' > "$OUT/rtt.tsv"
    port=9920
    for rep in $(seq "$REPS"); do
      if [ $((rep % 2)) -eq 1 ]; then ARMS="main fix sleep0"; else ARMS="sleep0 fix main"; fi
      for arm in $ARMS; do
        port=$((port + 1))
        start_server "$arm" "$port" 8192 "$OUT" "rtt-$arm-$rep"
        ms=$("$PY" "$E47/bench/rtt_client.py" "$port" "$N")
        kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
        printf '%s\t%s\t%s\n' "$arm" "$rep" "$ms" | tee -a "$OUT/rtt.tsv"
      done
    done
    ;;
  idle)
    REPS="$1"; SECS="$2"; OUT="$(absolute "$3")"; mkdir -p "$OUT"
    printf 'arm\trep\tcpu_seconds\twall_seconds\tcore_fraction\n' > "$OUT/idle.tsv"
    port=9930
    TICK=$(getconf CLK_TCK)
    for rep in $(seq "$REPS"); do
      if [ $((rep % 2)) -eq 1 ]; then ARMS="main fix sleep0"; else ARMS="sleep0 fix main"; fi
      for arm in $ARMS; do
        port=$((port + 1))
        start_server "$arm" "$port" 8192 "$OUT" "idle-$arm-$rep"
        sleep 3  # past start-up
        pid=$(pgrep -f "[p]lugrl_server.cli .*--port $port " | head -1)
        read -r u0 s0 < <(awk '{print $14, $15}' "/proc/$pid/stat")
        t0=$(date +%s.%N)
        sleep "$SECS"
        read -r u1 s1 < <(awk '{print $14, $15}' "/proc/$pid/stat")
        t1=$(date +%s.%N)
        kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
        python3 -c "c=(($u1+$s1)-($u0+$s0))/$TICK; w=$t1-$t0; print('$arm\t$rep\t%.3f\t%.3f\t%.4f' % (c, w, c/w))" | tee -a "$OUT/idle.tsv"
      done
    done
    ;;
  train)
    STEPS="$1"; OUT="$(absolute "$2")"; mkdir -p "$OUT"
    port=9910
    for arm in main fix; do
      port=$((port + 1))
      start_server "$arm" "$port" "$STEPS" "$OUT" "halfcheetah-$arm"
      (cd "$OUT" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
          --server-port "$port" --server-host 127.0.0.1 --num-envs 1 \
          --num-episodes $((STEPS / 1000 + 10)) --runner.replan-steps 1 --runner.seed 0 \
          --exp-name "e47-$arm" > "$OUT/client-$arm.log" 2>&1)
      echo "$arm client rc=$?"
      wait "$SERVER_PID"
      echo "$arm server rc=$?"
    done
    ;;
esac
echo "E47_${MODE}_DONE"
