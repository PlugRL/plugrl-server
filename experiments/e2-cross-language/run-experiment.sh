#!/usr/bin/env bash
# Start a dummy plugrl-server, point the dependency-free client at it, and
# report whether the exchange completed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_VENV="$HOME/.e2-server"
CLIENT_VENV="$HOME/.e2-client"
RESULTS="$HERE/results"
PORT="${E2_PORT:-8123}"
STEPS="${E2_STEPS:-20}"

GW=$(ip route | grep '^default' | tr -s ' ' | cut -d' ' -f3)
export http_proxy="http://$GW:7889" https_proxy="http://$GW:7889"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$RESULTS"

# The client venv holds msgpack and websockets and nothing else. No numpy, no
# PlugRL - if it can drive the server, the protocol needs neither.
if [ ! -d "$CLIENT_VENV" ]; then
    echo "creating client venv (msgpack + websockets only)..."
    uv venv --python 3.11 "$CLIENT_VENV" >/dev/null 2>&1
    VIRTUAL_ENV="$CLIENT_VENV" uv pip install --no-progress \
        "msgpack>=1.1.1,<2.0.0" "websockets>=15.0.1,<16.0.0" >/dev/null 2>&1
fi
echo "client venv contents:"
VIRTUAL_ENV="$CLIENT_VENV" uv pip list 2>/dev/null | sed 's/^/  /'

SERVER_LOG="$RESULTS/server.log"
CLIENT_LOG="$RESULTS/client.log"

echo
echo "starting server on port $PORT ..."
# A continuous 7-dim action, matching what dummy-v1 expects. The default
# policy is discrete, which the env rejects - a config mismatch, not a
# protocol one.
"$SERVER_VENV/bin/plugrl-run-server" \
    dummy-policy default dummy default \
    --policy.no-discrete --policy.action-dim 7 \
    --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 60); do
    grep -q "listening" "$SERVER_LOG" 2>/dev/null && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup:"
    tail -25 "$SERVER_LOG" | sed 's/^/  /'
    exit 1
fi
grep -i "listening" "$SERVER_LOG" | tail -1 | sed 's/^/  /'

echo
echo "running the dependency-free client ..."
"$CLIENT_VENV/bin/python" "$HERE/raw_client.py" \
    --host 127.0.0.1 --port "$PORT" --steps "$STEPS" 2>&1 | tee "$CLIENT_LOG"
STATUS=${PIPESTATUS[0]}

echo
if [ "$STATUS" -eq 0 ]; then
    echo "RESULT: the client completed the exchange without any PlugRL or numpy code."
else
    echo "RESULT: client failed (exit $STATUS). Server tail:"
    tail -20 "$SERVER_LOG" | sed 's/^/  /'
fi
exit "$STATUS"
