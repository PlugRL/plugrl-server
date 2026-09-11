#!/usr/bin/env bash
# Stand up a plugrl-server for the cross-language client experiment.
#
# The dummy policy and dummy algorithm exist for exactly this: exercising the
# protocol without a real model. That makes them the right target for asking
# whether the wire format can be spoken by something that is not this codebase.

set -uo pipefail

GW=$(ip route | grep '^default' | tr -s ' ' | cut -d' ' -f3)
export http_proxy="http://$GW:7889" https_proxy="http://$GW:7889"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export PATH="$HOME/.local/bin:$PATH"

CLEAN_GITCONFIG="$(mktemp)"
export GIT_CONFIG_GLOBAL="$CLEAN_GITCONFIG"
export GIT_CONFIG_SYSTEM=/dev/null
trap 'rm -f "$CLEAN_GITCONFIG"' EXIT

REPO=/mnt/d/75128/Desktop/plugrl-work
VENV="$HOME/.e2-server"

if [ ! -x "$VENV/bin/plugrl-run-server" ]; then
    echo "creating server venv..."
    uv venv --python 3.11 "$VENV"
    # Install from the local checkout so the pinned private protocol dep, which
    # WSL git cannot fetch, resolves to the local copy instead.
    VIRTUAL_ENV="$VENV" uv pip install --no-progress \
        "plugrl-protocol @ file://$REPO/plugrl-protocol" \
        "websockets>=15.0.1,<16.0.0" "tyro>=0.9.32,<0.10.0" \
        "loguru>=0.7.3,<0.8.0" "torch>=2.7.0,<2.8.0" \
        "safetensors>=0.5.3,<0.6.0" "packaging>=25.0,<26.0" \
        "python-dateutil>=2.9.0.post0,<3.0.0" "pyyaml>=6.0.3,<7.0.0" \
        "swanlab>=0.6.10,<0.7.0" "wandb>=0.22.0,<0.23.0" \
        "tensordict>=0.10.0,<0.11.0" "ray>=2.51.1,<3.0.0" \
        "tensorboard>=2.20.0,<3.0.0" "tqdm>=4.67.1,<5.0.0" \
        "rich>=13" || exit 1
    VIRTUAL_ENV="$VENV" uv pip install --no-progress --no-deps \
        "$REPO/plugrl-server" || exit 1
fi

echo "server venv: $VENV"
"$VENV/bin/python" -c "import plugrl_server, plugrl_protocol; print('imports ok')"
"$VENV/bin/plugrl-run-server" --help 2>&1 | head -20
