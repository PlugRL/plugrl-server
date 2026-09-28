#!/usr/bin/env bash
# E31: E28's two rising dppo-policy cells, run to 300 iterations. See PROTOCOL.md.
#
#   bash run.sh
#
# Both cells are E24's run_cell.sh, unchanged, as E28 ran them; only the
# number of iterations differs. The server runs this checkout's code through
# PYTHONPATH; the Pythons are the venvs beside it on guangzhao.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/results"
CELL="$ROOT/experiments/e24-coverage/run_cell.sh"
ITERS="${ITERS:-300}"
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
echo "client: $(git -C "$PLUGRL_ENV_CLIENT" rev-parse --short HEAD)"
echo "iters:  $ITERS"

BUFFER=1024 BATCH=512 PORT_BASE=9800 bash "$CELL" dppo-hopper dppo-policy hopper dppo hopper Hopper-v5 "$ITERS" "$R/dppo-hopper" > "$R/dppo-hopper.out" 2>&1 &
A=$!
sleep 20
BUFFER=1024 BATCH=512 PORT_BASE=9810 bash "$CELL" dppo-walker dppo-policy walker dppo walker Walker2d-v5 "$ITERS" "$R/dppo-walker" > "$R/dppo-walker.out" 2>&1 &
B=$!
wait "$A" "$B"

date '+end   %F %T'
for c in dppo-hopper dppo-walker; do
  printf '%-12s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E31_DONE
