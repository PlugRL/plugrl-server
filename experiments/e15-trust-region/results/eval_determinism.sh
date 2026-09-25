#!/usr/bin/env bash
# Is an evaluation deterministic when everything it is given is the same?
#
# E6 said no, in one sentence nobody followed up on: "both re-runs used
# --seed 0 on the server and --runner.seed 100 on the client and still
# produced different episodes, so the sampling path has a source of
# randomness those two seeds do not pin down."
#
# E14 and E15 then measured the same bit-identical pi0.5 at 35, 28 and 29 of
# 50 through the checkpoint path against 29, 29, 29 without one, and called it
# a property of checkpoint loading. If E6's observation is the same thing,
# that framing is wrong and every LIBERO number carries a spread nobody has
# quantified.
#
# This asks the cheap version of the question on HalfCheetah: same checkpoint,
# same two seeds, twice.
set -uo pipefail

W=/d/75128/Desktop/plugrl-work/plugrl-server/.claude/worktrees/fix-pi0-image-mask-batching
C=/d/75128/Desktop/plugrl-work/plugrl-env-client
CK="$W/experiments/e16-critic-restart/results/fpo/fpo-policy/halfcheetah-seed0/409600"
OUT=/c/Users/75128/.claude/jobs/1fea619d/tmp/eval-determinism
EPISODES=5
mkdir -p "$OUT"

run_eval() {   # $1 = label, $2 = port
  local label="$1" port="$2"
  (cd "$W" && "$W/.venv/Scripts/python.exe" -m plugrl_server.cli \
      fpo-policy default eval default \
      --port "$port" --seed 0 --policy.device cpu \
      --algo.policy-checkpoint-path "$CK" \
      --algo.num-episodes "$EPISODES" \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "$label" --overwrite \
      > "$OUT/server-$label.log" 2>&1 &)

  local waited=0
  until grep -q "is listening on" "$OUT/server-$label.log" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    [ "$waited" -ge 90 ] && { echo "$label: never listened"; return 1; }
  done

  (cd "$C" && "$C/.venv/Scripts/python.exe" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes "$EPISODES" \
      --runner.replan-steps 1 --runner.seed 100 \
      > "$OUT/client-$label.log" 2>&1)
  echo "$label done rc=$?"
}

run_eval run1 8950
run_eval run2 8951
echo EVAL_DETERMINISM_DONE
