#!/usr/bin/env bash
# E28: E24's five DPPO cells, run to 100 iterations. See PROTOCOL.md.
#
#   bash run.sh
#
# Every cell is E24's run_cell.sh, unchanged, writing into this experiment's
# results/. The server runs this checkout's code through PYTHONPATH; the
# Pythons are the venvs beside it on guangzhao, overridable from the
# environment. The three dppo-policy cells run first, then the two
# fpo-policy cells.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/results"
CELL="$ROOT/experiments/e24-coverage/run_cell.sh"
mkdir -p "$R"

export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/src"
export PLUGRL_ENV_CLIENT="${PLUGRL_ENV_CLIENT:-$ROOT/../plugrl-env-client}"
export PLUGRL_SERVER_PYTHON="${PLUGRL_SERVER_PYTHON:-$ROOT/../plugrl-server/.venv/bin/python}"
export PLUGRL_CLIENT_PYTHON="${PLUGRL_CLIENT_PYTHON:-$PLUGRL_ENV_CLIENT/.venv/bin/python}"
for py in "$PLUGRL_SERVER_PYTHON" "$PLUGRL_CLIENT_PYTHON"; do
  [ -x "$py" ] || { echo "error: no Python at $py" >&2; exit 1; }
done

date '+start %F %T'
echo "code:   $(git -C "$ROOT" rev-parse --short HEAD) (src $PYTHONPATH)"
echo "client: $(git -C "$PLUGRL_ENV_CLIENT" rev-parse --short HEAD) ($PLUGRL_CLIENT_PYTHON)"

echo "--- wave 1: dppo-policy"
BUFFER=1024 BATCH=512 PORT_BASE=9500 bash "$CELL" dppo-hopper dppo-policy hopper dppo hopper Hopper-v5 100 "$R/dppo-hopper" > "$R/dppo-hopper.out" 2>&1 &
A=$!
sleep 20
BUFFER=1024 BATCH=512 PORT_BASE=9510 bash "$CELL" dppo-walker dppo-policy walker dppo walker Walker2d-v5 100 "$R/dppo-walker" > "$R/dppo-walker.out" 2>&1 &
B=$!
sleep 20
BUFFER=1024 BATCH=512 PORT_BASE=9520 bash "$CELL" dppo-cheetah dppo-policy cheetah dppo cheetah HalfCheetah-v5 100 "$R/dppo-cheetah" > "$R/dppo-cheetah.out" 2>&1 &
C=$!
wait "$A" "$B" "$C"

echo "--- wave 2: fpo-policy"
PORT_BASE=9530 bash "$CELL" fpodppo-hopper fpo-policy default dppo hopper Hopper-v5 100 "$R/fpodppo-hopper" > "$R/fpodppo-hopper.out" 2>&1 &
D=$!
sleep 20
PORT_BASE=9540 bash "$CELL" fpodppo-walker fpo-policy default dppo walker Walker2d-v5 100 "$R/fpodppo-walker" > "$R/fpodppo-walker.out" 2>&1 &
E=$!
wait "$D" "$E"

date '+end   %F %T'
for c in dppo-hopper dppo-walker dppo-cheetah fpodppo-hopper fpodppo-walker; do
  printf '%-15s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E28_DONE
