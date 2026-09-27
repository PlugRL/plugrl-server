#!/usr/bin/env bash
# E33: fpo-policy under DPPO with E30's `all` structure, on all three MuJoCo
# tasks, 300 iterations. See PROTOCOL.md.
#
#   bash run.sh
#
# Every cell is E30's `all` arm unchanged: noise level 1.0 for sampling and
# the log-probability, four actions per inference with the client replanning
# every 4, twenty flow steps, three hidden layers of 512; buffer 1,024 and
# minibatch 512, so 4,096 environment steps per iteration. E30's run_cell.sh,
# unchanged. Hopper runs on seeds 3-5, which E30 never used; HalfCheetah and
# Walker2d on 0-2. The server runs this checkout's code through PYTHONPATH.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/results"
CELL="$ROOT/experiments/e30-fpo-dppo-factors/run_cell.sh"
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

WIDE="--algo.sampling-noise-level 1.0 --algo.logprob-noise-level 1.0"
STRUCTURE="--policy.action-horizon 4 --policy.flow-steps 20 --policy.hidden-dims 512 512 512"

date '+start %F %T'
echo "code:   $(git -C "$ROOT" rev-parse --short HEAD) (src $PYTHONPATH)"
echo "client: $(git -C "$PLUGRL_ENV_CLIENT" rev-parse --short HEAD)"
echo "iters:  $ITERS"

cell() {  # NAME ALGO_VARIANT TASK SEEDS PORT_BASE
  SEEDS="$4" BUFFER=1024 BATCH=512 REPLAN_OVERRIDE=4 PORT_BASE="$5" \
    EXTRA_ALGO_ARGS="$WIDE" EXTRA_POLICY_ARGS="$STRUCTURE" \
    bash "$CELL" "$1" fpo-policy default dppo "$2" "$3" "$ITERS" "$R/$1" > "$R/$1.out" 2>&1
}

cell fpo-cheetah cheetah HalfCheetah-v5 "0 1 2" 9900 &
A=$!
sleep 20
cell fpo-hopper hopper Hopper-v5 "3 4 5" 9910 &
B=$!
sleep 20
cell fpo-walker walker Walker2d-v5 "0 1 2" 9920 &
C=$!
wait "$A" "$B" "$C"

date '+end   %F %T'
for c in fpo-cheetah fpo-hopper fpo-walker; do
  printf '%-12s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E33_DONE
