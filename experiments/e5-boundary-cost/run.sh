#!/usr/bin/env bash
# What does the network boundary cost?
#
# E2 showed an env client can live anywhere and be written in anything. The
# obvious next question, and the one a reviewer will ask first, is what that
# costs per step. This measures it.
#
# The dummy algorithm's artificial sleeps are turned off, so the numbers are
# serialization plus transport plus deserialization and nothing else. Loopback
# only - this is the floor. Real cross-machine numbers need the cluster and
# will be higher; that measurement is E6.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP="$HERE/../e2-cross-language/plugrl_client.cpp"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
BIN="$HOME/.e5-client"
PORT="${E5_PORT:-8130}"
STEPS="${E5_STEPS:-200}"

echo "=== build ==="
g++ -std=c++17 -O2 -Wall -o "$BIN" "$CPP" 2>&1 | head -20
[ -x "$BIN" ] || { echo "BUILD FAILED"; exit 1; }
echo "  ok"

SERVER_LOG="$RESULTS/server.log"
# fake_inference_duration_sec=0 removes the deliberate sleep; a large
# fake_learn_freq keeps a learn phase from landing mid-measurement.
"$HOME/.e2-server/bin/plugrl-run-server" \
    dummy-policy default dummy default \
    --policy.no-discrete --policy.action-dim 7 \
    --algo.fake-inference-duration-sec 0 \
    --algo.fake-learn-duration-sec 0 \
    --algo.fake-learn-freq 1000000 \
    --algo.global-steps 100000 \
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
echo "  server up on $PORT"

: > "$RESULTS/summary.tsv"
printf 'batch\tcameras\timg\tbytes\tpack_ms\trtt_ms\tunpack_ms\n' >> "$RESULTS/summary.tsv"

echo
printf '%-28s %10s %9s %9s %9s\n' CONFIG PAYLOAD PACK RTT UNPACK
printf -- '-----------------------------------------------------------------------\n'

sweep() {
    local label="$1" batch="$2" img="$3" cams="$4"
    local out
    out=$("$BIN" 127.0.0.1 "$PORT" "$STEPS" "$batch" "$img" "$cams" 2>&1)
    local tsv
    tsv=$(printf '%s\n' "$out" | grep -oP '(?<=TSV\t).*' | tail -1)
    if [ -z "$tsv" ]; then
        printf '%-28s %s\n' "$label" "FAILED"
        printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
        return
    fi
    printf '%s\n' "$tsv" >> "$RESULTS/summary.tsv"
    local bytes pack rtt unpack
    bytes=$(printf '%s' "$tsv" | cut -f4)
    pack=$(printf '%s' "$tsv" | cut -f5)
    rtt=$(printf '%s' "$tsv" | cut -f6)
    unpack=$(printf '%s' "$tsv" | cut -f7)
    printf '%-28s %9sK %8sms %8sms %8sms\n' \
        "$label" "$((bytes / 1024))" "$pack" "$rtt" "$unpack"
}

# State only: the floor, what a proprioceptive robot would send.
sweep "states only (no camera)"   1 224 0
# One and two cameras at the resolutions dummy-v1 and LIBERO use.
sweep "1 cam @ 128px"             1 128 1
sweep "1 cam @ 224px"             1 224 1
sweep "2 cam @ 224px (dummy-v1)"  1 224 2
sweep "1 cam @ 448px"             1 448 1
# Batched: several envs multiplexed through one client.
sweep "2 cam @ 224px, batch 4"    4 224 2
sweep "2 cam @ 224px, batch 16"  16 224 2

printf -- '-----------------------------------------------------------------------\n'
echo "raw: $RESULTS/summary.tsv"
echo
echo "Loopback only. pack/unpack are the client's own CPU cost; rtt includes"
echo "the server's inference, which here is a no-op policy."
