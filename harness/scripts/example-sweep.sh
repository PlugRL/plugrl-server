#!/usr/bin/env bash
# A worked example, and the smoke test that matters: a matrix containing one
# cell of each kind.
#
# It should print three different verdicts - a combination that runs, one that
# fails, and one that could not be attempted. A run that does not is a sign
# the harness has stopped telling them apart, which is the one thing it exists
# to do.
#
# The two executables usually live in different environments: the server needs
# torch, the env client deliberately does not. Point at them explicitly.
#
#   PLUGRL_SERVER_BIN=/path/to/venv/bin/plugrl-run-server \
#   PLUGRL_CLIENT_BIN=/path/to/other/bin/plugrl-run-env-client \
#   bash scripts/example-sweep.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SWEEP_OUT:-sweep-out}"

SERVER_BIN="${PLUGRL_SERVER_BIN:-$(command -v plugrl-run-server || true)}"
CLIENT_BIN="${PLUGRL_CLIENT_BIN:-$(command -v plugrl-run-env-client || true)}"

if [ -z "$SERVER_BIN" ] || [ ! -x "$SERVER_BIN" ]; then
    echo "set PLUGRL_SERVER_BIN to an executable (got: ${SERVER_BIN:-<empty>})" >&2
    exit 1
fi
if [ -z "$CLIENT_BIN" ] || [ ! -x "$CLIENT_BIN" ]; then
    echo "set PLUGRL_CLIENT_BIN to an executable (got: ${CLIENT_BIN:-<empty>})" >&2
    exit 1
fi

export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# A policy must match the environment's action space, and environments
# disagree: CartPole takes one discrete action, the dummy env takes seven
# continuous ones. Hence the per-env overrides.
python -m plugrl_sweep.sweep \
    --envs dummy-v1 classic-v1 libero-v1 \
    --policies dummy-policy \
    --seeds 0 1 \
    --episodes 2 \
    --output-dir "$OUT" \
    --server-bin "$SERVER_BIN" \
    --client-bin "$CLIENT_BIN" \
    --client-timeout 120 \
    --policy-override discrete=false \
    --policy-override action_dim=7 \
    --env-policy-override classic-v1:discrete=true \
    --env-policy-override classic-v1:action_dim=2 \
    --env-policy-override classic-v1:action_horizon=1
