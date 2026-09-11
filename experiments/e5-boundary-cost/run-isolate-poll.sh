#!/usr/bin/env bash
# How much of the measured round-trip is the server's own polling, and how
# much is the boundary?
#
# Same client, same payload, only the server's scheduler poll interval
# changes. What the round-trip loses as the interval shrinks was never a cost
# of crossing the boundary.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.e5-client"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
STEPS="${E5_STEPS:-200}"

[ -x "$BIN" ] || { echo "build the client first (run.sh)"; exit 1; }

printf '%-22s %10s %10s\n' 'POLL INTERVAL' 'RTT mean' 'PACK mean'
printf -- '---------------------------------------------\n'

run_at() {
    local interval="$1" port="$2"
    local log="$RESULTS/poll-$interval.log"
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
    if ! kill -0 "$pid" 2>/dev/null; then
        printf '%-22s %s\n' "${interval}s" "server died"
        tail -5 "$log" | sed 's/^/    /'
        return
    fi

    local tsv
    tsv=$("$BIN" 127.0.0.1 "$port" "$STEPS" 1 224 2 2>&1 \
          | grep -oP '(?<=TSV\t).*' | tail -1)
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

    if [ -z "$tsv" ]; then
        printf '%-22s %s\n' "${interval}s" "client failed"
        return
    fi
    printf '%-22s %8sms %8sms\n' \
        "$(printf '%.3f' "$(echo "$interval * 1000" | bc -l)") ms" \
        "$(printf '%s' "$tsv" | cut -f6)" \
        "$(printf '%s' "$tsv" | cut -f5)"
}

run_at 0.001   8140   # the shipped default
run_at 0.0005  8141
run_at 0.0001  8142
run_at 0.00001 8143

printf -- '---------------------------------------------\n'
echo "Payload fixed at 2 cameras @ 224px (184 KiB), batch 1, loopback."
