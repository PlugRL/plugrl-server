#!/usr/bin/env bash
# E8: does turning the keepalive off actually stop the drops?
#
# The bug needs a learn step longer than the 20 s ping timeout. E6 used
# buffer_size=4096 - 16*ceil(4096/1024) = 64 gradient steps - and never
# tripped it. This raises the buffer until the learn step is long, holds
# everything else fixed, and runs the same configuration on both sides of
# the fix.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRV=/d/75128/Desktop/plugrl-work/plugrl-server
CLI=/d/75128/Desktop/plugrl-work/plugrl-env-client

LABEL="$1"        # before | after
BUFFER="${2:-32768}"
STEPS="${3:-32768}"
NENV="${4:-1}"
PORT="${5:-8611}"
# Gradient steps per learn are num_updates_per_batch * ceil(buffer/batch_size).
# Raising the epoch count rather than the buffer keeps the fill short while
# making the learn long, and it is the learn's duration - not the buffer's
# size - that the keepalive races against.
NUPD="${6:-400}"

OUT="$ROOT/$LABEL"
rm -rf "$OUT"; mkdir -p "$OUT"

# A client that dies on startup used to leave the server listening forever,
# and the next run then failed to bind. Kill it on the way out either way.
cleanup() {
  [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null
  # A client that dies on startup used to leave the server listening for
  # ever, and the next run then failed to bind. Take the port back by pid.
  for pid in $(netstat -ano 2>/dev/null | awk -v p=":$PORT" '$2 ~ p && $4=="LISTENING" {print $5}' | sort -u); do
    powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
  done
}
trap cleanup EXIT

echo "== $LABEL: buffer=$BUFFER steps=$STEPS envs=$NENV port=$PORT updates=$NUPD"
echo "   gradient steps per learn: $(( NUPD * ( (BUFFER + 1023) / 1024 ) ))"
echo "   server @ $(git -C "$SRV" rev-parse --abbrev-ref HEAD)  client @ $(git -C "$CLI" rev-parse --abbrev-ref HEAD)"

(cd "$SRV" && ./.venv/Scripts/python.exe -m plugrl_server.cli fpo-policy default fpo default \
   --port "$PORT" --policy.device cpu \
   --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
   --algo.num-updates-per-batch "$NUPD" \
   --no-show-progress-bar --no-show-metric-table \
   --checkpoint-base-dir "$OUT/ck" --exp-name e8 --overwrite > "$OUT/server.log" 2>&1) &
SRV_PID=$!

for i in $(seq 120); do
  grep -q "is listening" "$OUT/server.log" 2>/dev/null && break
  sleep 1
done
grep -q "is listening" "$OUT/server.log" || { echo "   server never listened"; tail -5 "$OUT/server.log"; exit 1; }

START=$(date +%s)
(cd "$CLI" && ./.venv/Scripts/python.exe -m plugrl_env_client.cli mujoco-v1 \
   --server-host 127.0.0.1 --server-port "$PORT" \
   --num-envs "$NENV" --num-episodes 100000 \
   --runner.replan-steps 1 --runner.seed 0 > "$OUT/client.log" 2>&1)
echo "   client exit $? after $(( $(date +%s) - START ))s"
wait $SRV_PID 2>/dev/null

echo "   keepalive timeouts (server): $(grep -c 'keepalive' "$OUT/server.log")"
echo "   1011 closes       (server): $(grep -c '1011' "$OUT/server.log")"
echo "   reconnects        (client): $(( $(grep -c 'Waiting for server' "$OUT/client.log") - 1 ))"
echo "   feedback-no-state (server): $(grep -c 'arrived with no step' "$OUT/server.log")"
