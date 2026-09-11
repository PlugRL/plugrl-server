#!/usr/bin/env bash
# A tighter poll loop uses more CPU per second under load - but it also
# completes more exchanges per second. CPU per second is the wrong unit; the
# question is what one exchange costs.
#
# So: fix the number of exchanges rather than the wall-clock window, and
# divide the server's CPU time by it.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.e5-client"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
STEPS="${E5_STEPS:-2000}"
TICKS_PER_SEC=$(getconf CLK_TCK)

[ -x "$BIN" ] || { echo "build the client first (run.sh)"; exit 1; }

cpu_seconds() {
    local pid="$1"
    [ -r "/proc/$pid/stat" ] || { echo 0; return; }
    awk -v t="$TICKS_PER_SEC" '{print ($14 + $15) / t}' "/proc/$pid/stat"
}

measure() {
    local interval="$1" port="$2"
    local log="$RESULTS/perex-$interval.log"

    "$HOME/.e2-server/bin/python" "$HERE/server_with_interval.py" "$interval" \
        dummy-policy default dummy default \
        --policy.no-discrete --policy.action-dim 7 \
        --algo.fake-inference-duration-sec 0 \
        --algo.fake-learn-duration-sec 0 \
        --algo.fake-learn-freq 10000000 \
        --algo.global-steps 100000000 \
        --port "$port" > "$log" 2>&1 &
    local pid=$!
    for _ in $(seq 60); do
        grep -q listening "$log" 2>/dev/null && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$pid" 2>/dev/null || { printf '%-14s server died\n' "$interval"; return; }

    local before after wall_start wall_end client_log client_rc
    client_log="$RESULTS/perex-client-$interval.log"
    before=$(cpu_seconds "$pid")
    wall_start=$(date +%s.%N)
    "$BIN" 127.0.0.1 "$port" "$STEPS" 1 224 2 > "$client_log" 2>&1
    client_rc=$?
    wall_end=$(date +%s.%N)
    after=$(cpu_seconds "$pid")
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    # A client that dies on connect still produces timings, and they look
    # spectacular. Refuse to report a number unless the run actually finished
    # the exchanges it was asked for.
    if [ "$client_rc" -ne 0 ] || \
       ! grep -q "completed $STEPS infer/action/feedback" "$client_log"; then
        printf '%-14s %s\n' \
            "$(printf '%.3f ms' "$(echo "$interval * 1000" | bc -l)")" \
            "DISCARDED - client did not complete $STEPS exchanges"
        tail -2 "$client_log" | sed 's/^/                 /'
        return
    fi

    local cpu wall rate per_ex
    cpu=$(echo "$after - $before" | bc -l)
    wall=$(echo "$wall_end - $wall_start" | bc -l)
    rate=$(echo "$STEPS / $wall" | bc -l)
    per_ex=$(echo "$cpu / $STEPS * 1000" | bc -l)

    printf '%-14s %8.2f s %9.2f s %10.0f/s %11.3f ms\n' \
        "$(printf '%.3f ms' "$(echo "$interval * 1000" | bc -l)")" \
        "$wall" "$cpu" "$rate" "$per_ex"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$interval" "$wall" "$cpu" "$rate" "$per_ex" >> "$RESULTS/perex.tsv"
}

echo "$STEPS exchanges, 2 cameras @ 224px, batch 1, loopback"
echo
printf '%-14s %10s %11s %12s %13s\n' \
    'POLL INTERVAL' 'WALL' 'SERVER CPU' 'THROUGHPUT' 'CPU/EXCHANGE'
printf -- '---------------------------------------------------------------------\n'
measure 0.001   8160
measure 0.0001  8161
measure 0.00001 8162
printf -- '---------------------------------------------------------------------\n'
