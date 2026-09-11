#!/usr/bin/env bash
# Build the C++ client and drive a real plugrl-server with it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
PORT="${E2_PORT:-8127}"
STEPS="${E2_STEPS:-120}"
BIN="$HOME/.e2-cpp-client"

echo "=== build ==="
echo "  g++ $(g++ -dumpversion), no third-party libraries"
g++ -std=c++17 -O2 -Wall -o "$BIN" "$HERE/plugrl_client.cpp" 2>&1 | head -30
[ -x "$BIN" ] || { echo "  BUILD FAILED"; exit 1; }
echo "  built: $(du -h "$BIN" | cut -f1)"
echo "  linked against:"
ldd "$BIN" | sed 's/^/    /'

SERVER_LOG="$RESULTS/cpp-server.log"
echo
echo "=== start server ==="
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
kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:"; tail -20 "$SERVER_LOG"; exit 1; }
grep -i listening "$SERVER_LOG" | tail -1 | sed 's/^/  /'

echo
echo "=== C++ client, $STEPS steps ==="
time "$BIN" 127.0.0.1 "$PORT" "$STEPS" 2>&1 | tee "$RESULTS/cpp-client.log"
STATUS=${PIPESTATUS[0]}

sleep 2
echo
echo "=== what the server recorded ==="
grep -iE "Step [0-9]+ /|total_connections|global_step" "$SERVER_LOG" | tail -6

echo
[ "$STATUS" -eq 0 ] \
    && echo "RESULT: a C++ client with no third-party libraries drove the server." \
    || { echo "RESULT: failed (exit $STATUS). Server tail:"; tail -15 "$SERVER_LOG"; }
exit "$STATUS"
