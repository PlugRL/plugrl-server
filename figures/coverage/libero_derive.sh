#!/usr/bin/env bash
# On qz103: derive the figure's LIBERO recording harness from E26's eval.sh.
#
#   bash libero_derive.sh
#
# Three changes and no others: a header line, output under $R/coverage, and
# the recorder writing videos. Refuses if either substitution would miss, and
# prints the diff.
set -euo pipefail
R=/home/gotham/tmp/plugrl
C=$R/coverage
mkdir -p "$C"
src=$R/e26/eval.sh
dst=$C/eval_record.sh
grep -c '^E=$R/e26$' "$src" | grep -qx 1
grep -c -- '--recorder.no-thread0-only \\$' "$src" | grep -qx 1
sed -e 's|^E=$R/e26$|E=$R/coverage|' \
    -e 's|      --recorder.no-thread0-only \\$|      --recorder.no-thread0-only --recorder.episode-freq 1 --recorder.record-video \\|' \
    -e '2i # Coverage-figure copy of E26'"'"'s eval.sh: output under coverage/, videos recorded.' \
    "$src" > "$dst"
chmod +x "$dst"
diff "$src" "$dst" || true
