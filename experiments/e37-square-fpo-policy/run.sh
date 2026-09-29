#!/usr/bin/env bash
# E37: fpo-policy on robomimic square from E35's behaviour-cloned start, under
# FPO (FPO++'s settings) and DPPO (`dppo square`). See PROTOCOL.md.
#
#   bash run.sh                                          # the registered run
#   FPO_ITERS=2 DPPO_ITERS=3 SEEDS=0 OUT=pilot bash run.sh   # a pilot
#
# The server runs this checkout's code through PYTHONPATH; the Pythons are
# the venvs beside it on guangzhao, the client's robomimic one.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/${OUT:-results}"
FPO_ITERS="${FPO_ITERS:-60}"
DPPO_ITERS="${DPPO_ITERS:-40}"
export SEEDS="${SEEDS:-0 1 2}"
export NPROC="${NPROC:-3}"
export BC="${BC:-$ROOT/../ckpt/e35-bc/400000}"
mkdir -p "$R"

export OMP_NUM_THREADS=1
export PYTHONPATH="$ROOT/src"
export PLUGRL_ENV_CLIENT="${PLUGRL_ENV_CLIENT:-$ROOT/../plugrl-env-client}"
export PLUGRL_SERVER_PYTHON="${PLUGRL_SERVER_PYTHON:-$ROOT/../plugrl-server/.venv/bin/python}"
export PLUGRL_CLIENT_PYTHON="${PLUGRL_CLIENT_PYTHON:-$PLUGRL_ENV_CLIENT/.venv-robomimic/bin/python}"
for py in "$PLUGRL_SERVER_PYTHON" "$PLUGRL_CLIENT_PYTHON"; do
  [ -x "$py" ] || { echo "error: no Python at $py" >&2; exit 1; }
done

date '+start %F %T'
echo "code:   $(git -C "$ROOT" rev-parse --short HEAD) (src $PYTHONPATH)"
echo "client: $(git -C "$PLUGRL_ENV_CLIENT" rev-parse --short HEAD)"
echo "iters:  fpo $FPO_ITERS, dppo $DPPO_ITERS  seeds: $SEEDS  clients per seed: $NPROC  out: $R"

PORT_BASE=9600 bash "$HERE/run_cell.sh" fpo-square fpo "$FPO_ITERS" "$R/fpo-square" > "$R/fpo-square.out" 2>&1 &
A=$!
sleep 30
PORT_BASE=9610 bash "$HERE/run_cell.sh" dppo-square dppo "$DPPO_ITERS" "$R/dppo-square" > "$R/dppo-square.out" 2>&1 &
B=$!
wait "$A" "$B"

date '+end   %F %T'
for c in fpo-square dppo-square; do
  printf '%-12s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E37_DONE
