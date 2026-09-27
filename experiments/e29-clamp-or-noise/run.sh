#!/usr/bin/env bash
# E29: fpo-policy under DPPO on Hopper, three arms at once. See PROTOCOL.md.
#
#   bash run.sh                          # the registered run: 100 iterations, seeds 0-2
#   ITERS=11 SEEDS=0 OUT=pilot bash run.sh  # a pilot
#
# Every arm is E28's fpodppo-hopper cell - fpo-policy, `dppo hopper`,
# Hopper-v5, buffer 4,096, the variant's minibatch - and differs only in the
# extra algorithm flags below. The server runs this checkout's code through
# PYTHONPATH; the Pythons are the venvs beside it on guangzhao.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/${OUT:-results}"
CELL="$HERE/run_cell.sh"
ITERS="${ITERS:-100}"
export SEEDS="${SEEDS:-0 1 2}"
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
echo "iters:  $ITERS  seeds: $SEEDS  out: $R"

arm() {  # NAME PORT_BASE EXTRA_ALGO_ARGS
  PORT_BASE="$2" EXTRA_ALGO_ARGS="$3" bash "$CELL" "$1" fpo-policy default dppo hopper \
    Hopper-v5 "$ITERS" "$R/$1" > "$R/$1.out" 2>&1
}

arm control 9600 "" &
A=$!
sleep 20
arm noclamp 9610 "--algo.logprob-clamp-max inf" &
B=$!
sleep 20
arm wide 9620 "--algo.sampling-noise-level 1.0 --algo.logprob-noise-level 1.0" &
C=$!
wait "$A" "$B" "$C"

date '+end   %F %T'
for a in control noclamp wide; do
  printf '%-8s %s\n' "$a" "$(grep -E 'failed seeds' "$R/$a.out" 2>/dev/null)"
done
echo E29_DONE
