#!/usr/bin/env bash
# The first learning curve this project has produced.
#
# Nothing in PlugRL had ever trained a policy: every tensorboard file on disk
# came from DummyAlgorithm, and no model weights existed anywhere in the git
# history of the four repositories. That is not a small gap for an RL
# framework, and it is the one this run closes.
#
# FPO on HalfCheetah-v5, three seeds, CPU only. The pairing is not arbitrary:
# HalfCheetah-v5 has a 17-dimensional observation and a 6-dimensional action,
# which are exactly FPOPolicyConfig's defaults, so the shipped policy points
# at it with no config surgery.
#
# One setting is NOT a default and has to be, because of how FPO decides to
# learn:
#
#   should_learn() is true when the rollout buffer is full, or when the run
#   has reached its last step. At the default buffer_size of 983040 a run of
#   under a million steps therefore learns exactly once, at the very end -
#   which produces a single point, not a curve. buffer_size=4096 gives one
#   update per 4096 environment steps.
#
# Usage:  bash run.sh [STEPS] [OUT_DIR]
set -euo pipefail

STEPS="${1:-500000}"
# Absolute, always. The server and client are each launched from their own
# directory, so a relative output path would be resolved against those and
# the redirect would land somewhere nobody looks - or fail.
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:-$HERE/results}"
case "$OUT" in
  /*) ;;
  *) OUT="$(pwd)/$OUT" ;;
esac
SEEDS="${SEEDS:-0 1 2}"
PORT_BASE="${PORT_BASE:-8600}"
BUFFER="${BUFFER:-4096}"

SERVER_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$OUT"

# Finding the two interpreters, in order of preference:
#
#   1. PLUGRL_SERVER_PYTHON / PLUGRL_CLIENT_PYTHON, if you set them;
#   2. a .venv beside each repository, Windows layout or POSIX layout;
#   3. whatever `python3`/`python` is on PATH - which is the right answer when
#      both packages are installed in one environment, as they can be.
#
# The env client's location is a guess (a sibling checkout) and guesses should
# be overridable, so PLUGRL_ENV_CLIENT exists.
CLIENT_DIR="${PLUGRL_ENV_CLIENT:-$SERVER_DIR/../plugrl-env-client}"
[ -d "$CLIENT_DIR" ] && CLIENT_DIR="$(cd "$CLIENT_DIR" && pwd)"

pick_python() {  # $1 = repo dir, $2 = override
  local repo="$1" override="${2:-}"
  if [ -n "$override" ]; then echo "$override"; return; fi
  for candidate in "$repo/.venv/Scripts/python.exe" "$repo/.venv/bin/python"; do
    if [ -x "$candidate" ]; then echo "$candidate"; return; fi
  done
  command -v python3 || command -v python
}

SERVER_PY="$(pick_python "$SERVER_DIR" "${PLUGRL_SERVER_PYTHON:-}")"
CLIENT_PY="$(pick_python "$CLIENT_DIR" "${PLUGRL_CLIENT_PYTHON:-}")"

# Fail now, with the reason, rather than after the first seed has silently
# written an empty log.
"$SERVER_PY" -c "import plugrl_server" 2>/dev/null || {
  echo "error: $SERVER_PY cannot import plugrl_server." >&2
  echo "       Set PLUGRL_SERVER_PYTHON to an interpreter that can." >&2
  exit 1
}
"$CLIENT_PY" -c "import plugrl_env_client.envs.mujoco.mujoco_env" 2>/dev/null || {
  echo "error: $CLIENT_PY cannot import the MuJoCo env." >&2
  echo "       Install it with:  uv sync --extra mujoco   (in plugrl-env-client)" >&2
  echo "       Or set PLUGRL_CLIENT_PYTHON / PLUGRL_ENV_CLIENT." >&2
  exit 1
}

for seed in $SEEDS; do
  port=$((PORT_BASE + seed))
  echo "=== seed $seed, $STEPS steps, port $port ==="

  # The server seed initialises the policy; the client seed initialises the
  # environment. Varying only one of them is not three seeds.
  (cd "$SERVER_DIR" && "$SERVER_PY" -m plugrl_server.cli \
      fpo-policy default fpo default \
      --port "$port" --seed "$seed" --policy.device cpu \
      --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
      --algo.save-interval 20 \
      --no-show-progress-bar --no-show-metric-table \
      --checkpoint-base-dir "$OUT" --exp-name "halfcheetah-seed$seed" --overwrite \
      > "$OUT/server-seed$seed.log" 2>&1 &)

  # Wait for the server to say it is listening, rather than sleeping a guessed
  # number of seconds. Probing the TCP port would also work and be shorter,
  # but a bare connection is not a WebSocket handshake, so the server logs
  # `InvalidMessage: did not receive a valid HTTP request` for every probe -
  # an exception in the log of every run, caused by the harness watching it.
  for _ in $(seq 120); do
    grep -q "is listening on" "$OUT/server-seed$seed.log" 2>/dev/null && break
    sleep 1
  done

  # num-episodes is a ceiling the server's global-steps reaches first; each
  # HalfCheetah episode is 1000 steps.
  (cd "$CLIENT_DIR" && "$CLIENT_PY" -m plugrl_env_client.cli mujoco-v1 \
      --server-port "$port" --server-host 127.0.0.1 \
      --num-envs 1 --num-episodes $((STEPS / 1000 + 10)) \
      --runner.replan-steps 1 --runner.seed "$seed" \
      > "$OUT/client-seed$seed.log" 2>&1)

  echo "    seed $seed done"
done

echo "all seeds finished; results under $OUT"
