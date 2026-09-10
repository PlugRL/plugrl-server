#!/usr/bin/env bash
# Set up the WSL side and run the footprint measurement.
set -uo pipefail

GW=$(ip route | grep '^default' | tr -s ' ' | cut -d' ' -f3)
export http_proxy="http://$GW:7889" https_proxy="http://$GW:7889"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export PATH="$HOME/.local/bin:$PATH"

# Ubuntu's git is built against gnutls and refuses this machine's global
# http.sslBackend=openssl. Use a clean config rather than editing the user's.
CLEAN_GITCONFIG="$(mktemp)"
export GIT_CONFIG_GLOBAL="$CLEAN_GITCONFIG"
export GIT_CONFIG_SYSTEM=/dev/null
trap 'rm -f "$CLEAN_GITCONFIG"' EXIT

export UV_CACHE_DIR="$HOME/.cache/uv"
export E1_WORK="$HOME/.e1-footprint"

echo "uv = $(uv --version 2>&1)"
echo
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-footprint.sh"
