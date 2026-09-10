#!/usr/bin/env bash
# E5 recommended lowering the scheduler poll interval for a 3x round-trip win,
# with a caveat: a tighter loop should burn more CPU when nothing is happening.
# That caveat is unverified, and recommending a change while leaving its cost
# unmeasured is not much of a recommendation. So measure it.
#
# Two states matter:
#   idle  - server up, no client connected
#   busy  - one client driving it as fast as it can
#
# Measured as CPU seconds consumed over a fixed wall-clock window, read from
# /proc/<pid>/stat so nothing depends on top's sampling.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.e5-client"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
WINDOW="${E5_WINDOW:-15}"
TICKS_PER_SEC=$(getconf CLK_TCK)

[ -x "$BIN" ] || { echo "build the client first (run.sh)"; exit 1; }

cpu_seconds() {
    # utime + stime from /proc/<pid>/stat, fields 14 and 15.
    local pid="$1"
    [ -r "/proc/$pid/stat" ] || { echo 0; return; }
    awk -v t="$TICKS_PER_SEC" '{print ($14 + $15) / t}' "/proc/$pid/stat"
}

measure() {
    local interval="$1" port="$2" mode="$3"
    local log="$RESULTS/idle-$interval-$mode.log"

    "$HOME/.e2-server/bin/python" "$HERE/server_with_interval.py" "$interval" \
        dummy-policy default dummy default \
        --policy.no-discrete --policy.action-dim 7 \
        --algo.fake-inference-duration-sec 0 \
        --algo.fake-learn-duration-sec 0 \
        --algo.fake-learn-freq 1000000 \
        --algo.global-steps 100000 \
        --port "$port" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 60); do
        grep -q listening "$log" 2>/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$pid" 2>/dev/null || { echo "server died"; return; }

    local client_pid=""
    if [ "$mode" = busy ]; then
        "$BIN" 127.0.0.1 "$port" 1000000 1 224 2 >/dev/null 2>&1 &
        client_pid=$!
        sleep 2   # let the exchange settle before the window opens
    fi

    local before after used pct
    before=$(cpu_seconds "$pid")
    sleep "$WINDOW"
    after=$(cpu_seconds "$pid")
    used=$(echo "$after - $before" | bc -l)
    pct=$(echo "$used / $WINDOW * 100" | bc -l)

    [ -n "$client_pid" ] && { kill "$client_pid" 2>/dev/null; wait "$client_pid" 2>/dev/null; }
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    printf '%-14s %-6s %9.2f s %9.1f %%\n' \
        "$(printf '%.3f ms' "$(echo "$interval * 1000" | bc -l)")" \
        "$mode" "$used" "$pct"
}

echo "server CPU over a ${WINDOW}s window (one core = 100%)"
echo
printf '%-14s %-6s %11s %11s\n' 'POLL INTERVAL' 'STATE' 'CPU USED' 'OF ONE CORE'
printf -- '------------------------------------------------------\n'
measure 0.001   8150 idle
measure 0.0001  8151 idle
measure 0.00001 8152 idle
printf -- '------------------------------------------------------\n'
measure 0.001   8153 busy
measure 0.0001  8154 busy
printf -- '------------------------------------------------------\n'
