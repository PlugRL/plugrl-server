#!/usr/bin/env bash
# E38: gaussian-policy under ppo - CleanRL's ppo_continuous_action.py - on the
# three MuJoCo tasks, 488 iterations. See PROTOCOL.md.
#
#   bash run.sh
#
# Every cell is CleanRL's file with its defaults: a rollout of 2,048 steps,
# minibatches of 64, 488 iterations (999,424 steps: CleanRL runs
# 1,000,000 // 2,048 = 488). One environment per seed, as CleanRL's
# num_envs=1. E30's run_cell.sh, unchanged, given the task's
# dimensions through EXTRA_POLICY_ARGS. The server runs this checkout's code
# through PYTHONPATH.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$HERE/results"
CELL="$ROOT/experiments/e30-fpo-dppo-factors/run_cell.sh"
ITERS="${ITERS:-488}"
SEEDS="${SEEDS:-0 1 2}"
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
echo "iters:  $ITERS  seeds: $SEEDS"

cell() {  # NAME TASK OBS_DIM ACTION_DIM PORT_BASE
  SEEDS="$SEEDS" BUFFER=2048 BATCH=64 PORT_BASE="$5" \
    EXTRA_POLICY_ARGS="--policy.obs-dim $3 --policy.action-dim $4" \
    bash "$CELL" "$1" gaussian-policy default ppo default "$2" "$ITERS" "$R/$1" \
    > "$R/$1.out" 2>&1
}

cell ppo-cheetah HalfCheetah-v5 17 6 9940 &
A=$!
sleep 20
cell ppo-hopper Hopper-v5 11 3 9950 &
B=$!
sleep 20
cell ppo-walker Walker2d-v5 17 6 9960 &
C=$!
wait "$A" "$B" "$C"

date '+end   %F %T'
for c in ppo-cheetah ppo-hopper ppo-walker; do
  printf '%-12s %s\n' "$c" "$(grep -E 'failed seeds' "$R/$c.out" 2>/dev/null)"
done
echo E38_DONE
