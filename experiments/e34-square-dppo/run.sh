#!/usr/bin/env bash
# E34: dppo-policy under DPPO on robomimic square, DPPO's own fine-tuning of
# its released checkpoint. See PROTOCOL.md.
#
#   bash run.sh                                   # the registered run
#   ITERS=3 SEEDS=0 OUT=pilot bash run.sh         # a pilot
#
# The server runs this checkout's code through PYTHONPATH; the Pythons are
# the venvs beside it on guangzhao, the client's robomimic one
# (.venv-robomimic: robomimic v0.4.0, robosuite 1.4.1, mujoco 2.3.7).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/${OUT:-results}"
ITERS="${ITERS:-40}"
export SEEDS="${SEEDS:-0 1 2}"
export NPROC="${NPROC:-6}"
export CKPT="${CKPT:-$ROOT/../ckpt/dppo-square/state_8000.pt}"
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
echo "iters:  $ITERS  seeds: $SEEDS  clients per seed: $NPROC  out: $R"

PORT_BASE=9500 bash "$HERE/run_cell.sh" dppo-square "$ITERS" "$R/dppo-square" > "$R/dppo-square.out" 2>&1

date '+end   %F %T'
printf '%-12s %s\n' dppo-square "$(grep -E 'failed seeds' "$R/dppo-square.out" 2>/dev/null)"
echo E34_DONE
