#!/usr/bin/env bash
# E45: what the environment side must install to exchange training data with
# its trainer, for one system at a time, in a fresh venv with a cold cache.
#
#   bash measure.sh NAME OUT_DIR
#
# Appends one row to OUT_DIR/footprint.tsv:
#   name  install_rc  packages  bytes  nvidia_packages  torch  entry_import  seconds
# and keeps the install log, `uv pip list` and the import's stderr. The venv
# and its cache are deleted afterwards. `bytes` is the venv, which does not
# include the interpreter: every system here uses one of uv's managed Pythons.
set -uo pipefail

NAME="$1"
OUT="$2"
UV="${UV:-$HOME/.local/bin/uv}"
WORK="${WORK:-$HOME/zuogou/plugrl/e45-work}"
mkdir -p "$WORK" "$OUT"
V="$WORK/venv-$NAME"
export UV_CACHE_DIR="$WORK/cache-$NAME"
rm -rf "$V" "$UV_CACHE_DIR"

PLUGRL_ENV_CLIENT="git+https://github.com/PlugRL/plugrl-env-client@3ea06099143f4764e1678d17e5463049c8aeffbb"
OPENPI="git+https://github.com/Physical-Intelligence/openpi.git@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client"

case "$NAME" in
  plugrl-env-client)  # PlugRL's Python env client, no environment extras
    PY=3.11; SPEC=("plugrl-env-client @ $PLUGRL_ENV_CLIENT")
    ENTRY="import plugrl_env_client.cli" ;;
  plugrl-minimal)     # what plugrl-protocol's examples/raw_client.py imports
    PY=3.11; SPEC=(msgpack websockets)
    ENTRY="import msgpack, websockets.sync.client" ;;
  rllib-rllink)       # RLlib's external-env client, as its reference client imports it
    PY=3.11; SPEC=("ray[rllib]==2.58.0" torch)
    ENTRY="import ray.rllib.examples.envs.classes.utils.dummy_external_client" ;;
  lerobot-hilserl)    # LeRobot's HIL-SERL actor
    PY=3.12; SPEC=("lerobot[hilserl]==0.6.1")
    ENTRY="import lerobot.rl.actor" ;;
  dm-env-rpc)         # DeepMind's env protocol, environment side
    PY=3.11; SPEC=("dm-env-rpc==1.1.7")
    ENTRY="from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc" ;;
  openpi-client)      # openpi's policy client, inference only
    PY=3.11; SPEC=("openpi-client @ $OPENPI")
    ENTRY="from openpi_client import websocket_client_policy" ;;
  *) echo "unknown system $NAME" >&2; exit 2 ;;
esac

start=$(date +%s.%N)
"$UV" venv -q --python "$PY" "$V"
"$UV" pip install --python "$V/bin/python" "${SPEC[@]}" > "$OUT/$NAME.install.log" 2>&1
rc=$?
end=$(date +%s.%N)

"$UV" pip list --python "$V/bin/python" > "$OUT/$NAME.pip-list.txt" 2> /dev/null
# The rows under the dashed line; uv's "Using Python ..." note goes to stderr.
packages=$(awk 'seen { n++ } /^-+ +-+/ { seen = 1 } END { print n + 0 }' "$OUT/$NAME.pip-list.txt")
bytes=$(du -sb "$V" | cut -f1)
nvidia=$(grep -c -i "^nvidia-" "$OUT/$NAME.pip-list.txt" || true)
torch=$("$V/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo none)
if "$V/bin/python" -c "$ENTRY" > /dev/null 2> "$OUT/$NAME.entry.log"; then entry=ok; else entry=FAIL; fi
seconds=$(python3 -c "print(f'{$end - $start:.1f}')")

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$NAME" "$rc" "$packages" "$bytes" "$nvidia" "$torch" "$entry" "$seconds" | tee -a "$OUT/footprint.tsv"
rm -rf "$V" "$UV_CACHE_DIR"
