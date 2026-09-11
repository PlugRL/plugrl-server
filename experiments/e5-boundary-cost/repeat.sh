#!/usr/bin/env bash
# Five repetitions, then a median. One sample was not enough: the first run
# said 0.01 ms was worse than 0.1 ms, and repeats said the opposite. A single
# sample also hid a harness fault - a client that failed on connect reported
# 118879 exchanges/s, and nothing caught it. run-cpu-per-exchange.sh now
# discards runs that did not complete; this drives it enough times for the
# medians to mean something.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$HERE/results"; mkdir -p "$RESULTS"
: > "$RESULTS/perex.tsv"

REPS="${E5_REPS:-5}"
for i in $(seq "$REPS"); do
    echo "===== repetition $i / $REPS ====="
    bash "$HERE/run-cpu-per-exchange.sh" 2>&1 | tail -6
    echo
done

echo "===== medians over valid runs ====="
python3 - "$RESULTS/perex.tsv" <<'PY'
import statistics
import sys
from collections import defaultdict

rows = defaultdict(list)
with open(sys.argv[1]) as fh:
    for line in fh:
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        interval, wall, cpu, rate, per_ex = parts
        rows[float(interval)].append((float(wall), float(rate), float(per_ex)))

print(f"{'POLL INTERVAL':<14} {'n':>3} {'THROUGHPUT':>14} {'CPU/EXCHANGE':>16}")
print("-" * 52)
for interval in sorted(rows, reverse=True):
    samples = rows[interval]
    rate = statistics.median(s[1] for s in samples)
    per_ex = statistics.median(s[2] for s in samples)
    spread = ""
    if len(samples) > 1:
        lo = min(s[1] for s in samples)
        hi = max(s[1] for s in samples)
        spread = f"  ({lo:.0f}-{hi:.0f})"
    print(f"{interval * 1000:>10.3f} ms {len(samples):>3} "
          f"{rate:>10.0f}/s {per_ex:>13.3f} ms{spread}")
PY
