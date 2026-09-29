#!/usr/bin/env bash
# E45's registered run: every system, one at a time, in PROTOCOL.md's order.
#   bash run.sh [OUT_DIR]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/results}"
mkdir -p "$OUT"
printf 'name\tinstall_rc\tpackages\tbytes\tnvidia_packages\ttorch\tentry_import\tseconds\n' > "$OUT/footprint.tsv"
echo "uv: $("${UV:-$HOME/.local/bin/uv}" --version)  host: $(hostname)  $(uname -srm)"
date '+start %F %T'
for name in plugrl-minimal plugrl-env-client dm-env-rpc openpi-client rllib-rllink lerobot-hilserl; do
  bash "$HERE/measure.sh" "$name" "$OUT"
done
date '+end   %F %T'
echo RUN_DONE
