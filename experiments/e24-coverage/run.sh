#!/usr/bin/env bash
# E24: the six cells, in two waves. See PROTOCOL.md.
#
#   bash run.sh
#
# Wave 1 runs the long FPO cell beside the two short fpo-policy DPPO cells -
# nine server-client pairs. Wave 2 runs the three dppo-policy cells. Each cell
# writes to results/<cell>/, and its own output to results/<cell>.out.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/results"
mkdir -p "$R"
CELL="$HERE/run_cell.sh"

date '+start %F %T'

echo "--- wave 1"
PORT_BASE=9300 bash "$CELL" fpo-walker fpo-policy default fpo default Walker2d-v5 100 > "$R/fpo-walker.out" 2>&1 &
W1=$!
sleep 20
PORT_BASE=9310 bash "$CELL" fpodppo-hopper fpo-policy default dppo hopper Hopper-v5 20 > "$R/fpodppo-hopper.out" 2>&1 &
W2=$!
sleep 20
PORT_BASE=9320 bash "$CELL" fpodppo-walker fpo-policy default dppo walker Walker2d-v5 20 > "$R/fpodppo-walker.out" 2>&1 &
W3=$!
wait "$W2" "$W3"
echo "short cells of wave 1 done at $(date '+%T')"

echo "--- wave 2"
BUFFER=1024 BATCH=512 PORT_BASE=9330 bash "$CELL" dppo-cheetah dppo-policy cheetah dppo cheetah HalfCheetah-v5 20 > "$R/dppo-cheetah.out" 2>&1 &
D1=$!
sleep 20
BUFFER=1024 BATCH=512 PORT_BASE=9340 bash "$CELL" dppo-hopper dppo-policy hopper dppo hopper Hopper-v5 20 > "$R/dppo-hopper.out" 2>&1 &
D2=$!
sleep 20
BUFFER=1024 BATCH=512 PORT_BASE=9350 bash "$CELL" dppo-walker dppo-policy walker dppo walker Walker2d-v5 20 > "$R/dppo-walker.out" 2>&1 &
D3=$!
wait "$W1" "$D1" "$D2" "$D3"

date '+end   %F %T'
for c in fpo-walker fpodppo-hopper fpodppo-walker dppo-cheetah dppo-hopper dppo-walker; do
  printf '%-15s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E24_DONE
