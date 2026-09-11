#!/usr/bin/env bash
# Install the real plugrl-env-client, to establish a working baseline. If the
# real client cannot connect either, the problem is the server or the harness,
# not the dependency-free client.
set -uo pipefail

GW=$(ip route | grep '^default' | tr -s ' ' | cut -d' ' -f3)
export http_proxy="http://$GW:7889" https_proxy="http://$GW:7889"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export PATH="$HOME/.local/bin:$PATH"

REPO=/mnt/d/75128/Desktop/plugrl-work
VENV="$HOME/.e2-envclient"

if [ ! -x "$VENV/bin/plugrl-run-env-client" ]; then
    uv venv --python 3.11 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install --no-progress \
        "plugrl-protocol @ file://$REPO/plugrl-protocol" \
        "loguru>=0.7.3,<0.8.0" "gymnasium>=1.2.0,<2.0.0" \
        "dm-tree>=0.1.9,<0.2.0" "websockets>=15.0.1,<16.0.0" \
        "msgpack>=1.1.1,<2.0.0" "tyro>=0.9.31,<0.10.0" \
        "pandas>=2.3.3,<3.0.0" "imageio[ffmpeg]>=2.37.2,<3.0.0" || exit 1
    VIRTUAL_ENV="$VENV" uv pip install --no-progress --no-deps \
        "$REPO/plugrl-env-client" || exit 1
fi

echo "env-client venv: $VENV"
"$VENV/bin/python" -c "import plugrl_env_client; print('import ok')"
