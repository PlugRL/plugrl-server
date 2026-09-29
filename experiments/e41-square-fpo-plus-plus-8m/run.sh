#!/usr/bin/env bash
# E41: E39 resumed and run on to FPO++'s own 8M steps (167 iterations of
# 48,000). See PROTOCOL.md.
#
#   bash run.sh                                   # the registered run
#   ITERS=102 SEEDS=0 OUT=pilot bash run.sh       # a pilot: two iterations
#
# E39 is the run directory of E39's registered run on guangzhao.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/${OUT:-results}"
ITERS="${ITERS:-167}"
export SEEDS="${SEEDS:-0 1 2}"
export NPROC="${NPROC:-3}"
export E39="${E39:-$ROOT/../e39-reg/experiments/e39-square-fpo-plus-plus/results/fpopp-square/fpo/fpo-policy}"
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
echo "iters:  to $ITERS  seeds: $SEEDS  clients per seed: $NPROC  out: $R"

PORT_BASE=9700 bash "$HERE/run_cell.sh" fpopp-square-8m "$ITERS" "$R/fpopp-square-8m" \
  > "$R/fpopp-square-8m.out" 2>&1

date '+end   %F %T'
printf '%-16s %s\n' fpopp-square-8m "$(grep -E 'failed seeds' "$R/fpopp-square-8m.out" 2>/dev/null)"
echo E41_DONE
