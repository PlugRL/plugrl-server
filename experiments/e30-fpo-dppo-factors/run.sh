#!/usr/bin/env bash
# E30: fpo-policy under DPPO on Hopper, five arms in two waves. See PROTOCOL.md.
#
#   bash run.sh                             # the registered run: 100 iterations, seeds 0-2
#   ITERS=11 SEEDS=0 OUT=pilot bash run.sh  # a pilot
#
# Every arm is E29's `wide` - fpo-policy, `dppo hopper`, Hopper-v5, noise
# level 1.0 for sampling and log-probability - and changes one of the three
# ways fpo-policy still differs from dppo-policy, or all three:
#
#   base     nothing more
#   chunk4   four actions per inference (dppo-policy's chunk), client replans
#            every 4; buffer 1,024 and minibatch 512 keep 4,096 environment
#            steps and two minibatches per iteration, as for dppo-policy
#   steps20  twenty flow steps (dppo-policy's twenty denoising steps)
#   net512   hidden layers 512 x 3 (dppo-policy's MLP width and depth)
#   all      all three
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

WIDE="--algo.sampling-noise-level 1.0 --algo.logprob-noise-level 1.0"
CHUNK="--policy.action-horizon 4"
STEPS="--policy.flow-steps 20"
NET="--policy.hidden-dims 512 512 512"

date '+start %F %T'
echo "code:   $(git -C "$ROOT" rev-parse --short HEAD) (src $PYTHONPATH)"
echo "client: $(git -C "$PLUGRL_ENV_CLIENT" rev-parse --short HEAD)"
echo "iters:  $ITERS  seeds: $SEEDS  out: $R"

arm() {  # NAME PORT_BASE POLICY_FLAGS [chunk]
  local buffer=4096 batch="" replan=1
  if [ "${4:-}" = chunk ]; then buffer=1024 batch=512 replan=4; fi
  BUFFER=$buffer BATCH=$batch REPLAN_OVERRIDE=$replan PORT_BASE="$2" \
    EXTRA_ALGO_ARGS="$WIDE" EXTRA_POLICY_ARGS="$3" \
    bash "$CELL" "$1" fpo-policy default dppo hopper Hopper-v5 "$ITERS" "$R/$1" > "$R/$1.out" 2>&1
}

echo "--- wave 1"
arm base 9700 "" &
A=$!
sleep 15
arm chunk4 9710 "$CHUNK" chunk &
B=$!
sleep 15
arm steps20 9720 "$STEPS" &
C=$!
wait "$A" "$B" "$C"

echo "--- wave 2"
arm net512 9730 "$NET" &
D=$!
sleep 15
arm all 9740 "$CHUNK $STEPS $NET" chunk &
E=$!
wait "$D" "$E"

date '+end   %F %T'
for a in base chunk4 steps20 net512 all; do
  printf '%-8s %s\n' "$a" "$(grep -E 'failed seeds' "$R/$a.out" 2>/dev/null)"
done
echo E30_DONE
