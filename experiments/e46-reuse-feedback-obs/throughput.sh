#!/usr/bin/env bash
# Laptop -> guangzhao TCP throughput over Tailscale, REPS times, MIB each.
# E43's throughput.sh, with E46's paths.
#   bash throughput.sh REPS MIB OUT_TSV
set -uo pipefail
REPS="$1"
MIB="$2"
TSV="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${REMOTE:-guangzhao-memeseeks}"
TS_IP="${TS_IP:-100.75.226.89}"
LOCAL_PY="${LOCAL_PY:?set LOCAL_PY}"
R_SCRIPT='~/zuogou/plugrl/e46/bench/throughput.py'
printf 'rep\tmib\tseconds\tmb_per_s\n' > "$TSV"
for rep in $(seq "$REPS"); do
  port=$((9960 + rep))
  ssh "$REMOTE" "(nohup python3 $R_SCRIPT sink $port > /tmp/e46tp_$port.log 2>&1 &); for i in \$(seq 50); do grep -q listening /tmp/e46tp_$port.log && exit 0; sleep 0.1; done; exit 1" || { echo "sink failed" >&2; continue; }
  "$LOCAL_PY" "$HERE/throughput.py" send "$TS_IP" "$port" "$MIB"
  sleep 1
  line=$(ssh "$REMOTE" "tail -1 /tmp/e46tp_$port.log")
  printf '%s\t%s\n' "$rep" "$line" | tee -a "$TSV"
done
