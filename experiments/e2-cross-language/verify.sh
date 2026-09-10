#!/usr/bin/env bash
# Did the server treat the dependency-free client as a real env client, or did
# it merely tolerate the messages? Run a longer session and read the server's
# own accounting.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
PORT="${E2_PORT:-8126}"
STEPS="${E2_STEPS:-120}"

SERVER_LOG="$RESULTS/verify-server.log"

"$HOME/.e2-server/bin/plugrl-run-server" \
    dummy-policy default dummy default \
    --policy.no-discrete --policy.action-dim 7 \
    --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 60); do
    grep -q listening "$SERVER_LOG" 2>/dev/null && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
done

echo "=== $STEPS exchanges from the dependency-free client ==="
time "$HOME/.e2-client/bin/python" "$HERE/raw_client.py" \
    --host 127.0.0.1 --port "$PORT" --steps "$STEPS" 2>&1 | tail -5

sleep 2
echo
echo "=== what the server recorded ==="
grep -iE "connection|step|collect|learn|metric|episode" "$SERVER_LOG" \
    | grep -viE "checkpoint manager|version|Config:" | tail -18
