#!/usr/bin/env bash
# E27: the three robomimic cells, concurrently. See PROTOCOL.md.
#
#   OMP_NUM_THREADS=1 bash run.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/results"
mkdir -p "$R"
CELL="$HERE/run_cell.sh"

date '+start %F %T'
PORT_BASE=9400 bash "$CELL" fpo-rm fpo-policy default fpo default 20 > "$R/fpo-rm.out" 2>&1 &
A=$!
sleep 20
PORT_BASE=9410 BATCH=256 bash "$CELL" fpodppo-rm fpo-policy default dppo default 20 > "$R/fpodppo-rm.out" 2>&1 &
B=$!
sleep 20
PORT_BASE=9420 BUFFER=1024 BATCH=256 bash "$CELL" dppo-rm dppo-policy default dppo default 20 > "$R/dppo-rm.out" 2>&1 &
C=$!
wait "$A" "$B" "$C"
date '+end   %F %T'
for c in fpo-rm fpodppo-rm dppo-rm; do
  printf '%-12s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E27_DONE
