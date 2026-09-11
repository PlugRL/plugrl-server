#!/usr/bin/env bash
# Bootstrap the E1 install-layer experiment inside WSL and run it.
#
# WSL2 has no working DNS here, so everything goes through the proxy running on
# the Windows host, reachable at the default gateway.

set -uo pipefail

GW=$(ip route | grep '^default' | tr -s ' ' | cut -d' ' -f3)
if [ -z "$GW" ]; then
    echo "no default route; cannot reach the host proxy" >&2
    exit 1
fi
export http_proxy="http://$GW:7889"
export https_proxy="http://$GW:7889"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
echo "proxy = $https_proxy"

export PATH="$HOME/.local/bin:$PATH"

# This machine's global git config sets http.sslBackend=openssl, but the git
# shipped with Ubuntu is built against gnutls and refuses to start with it.
# Use a clean config for everything below rather than editing the user's.
CLEAN_GITCONFIG="$(mktemp)"
export GIT_CONFIG_GLOBAL="$CLEAN_GITCONFIG"
export GIT_CONFIG_SYSTEM=/dev/null
trap 'rm -f "$CLEAN_GITCONFIG"' EXIT
echo "git   = $(git --version 2>&1)"

if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv    = $(uv --version 2>&1)"

# Keep venvs and caches on the Linux filesystem. Building against /mnt/d is
# slow enough to distort the timings and can break packages that mmap files.
export UV_CACHE_DIR="$HOME/.cache/uv"
export E1_WORK="$HOME/.e1-install"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run-install.sh"
