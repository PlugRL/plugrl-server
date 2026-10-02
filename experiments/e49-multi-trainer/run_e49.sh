#!/usr/bin/env bash
# E49's 12 registered runs on guangzhao, one at a time (PROTOCOL.md).
#   bash run_e49.sh OUT_DIR     # e.g. ~/zuogou/plugrl/e49/registered
set -uo pipefail
OUT="${1:?OUT_DIR}"
BASE="$HOME/zuogou/plugrl"
RUN="$BASE/plugrl-bridges/examples/run_bridged.sh"
mkdir -p "$OUT"
(cd "$BASE/plugrl-bridges" && find src examples -name "*.py" -o -name "*.sh" -o -name "*.yaml" | sort | xargs sha256sum) > "$OUT/bridges-src.sha256"
uptime | tee "$OUT/load.txt"
# The fourth field is the step budget: E44's 102,400 for Pendulum, E43's
# 409,600 for HalfCheetah, and 409,600 for RLinf x Pendulum (PROTOCOL.md).
for cell in "rlinf pendulum 409600" "sb3 pendulum 102400" "sb3 halfcheetah 409600" "cleanrl halfcheetah 409600"; do
  set -- $cell
  for seed in 0 1 2; do
    name="$1-$2-seed$seed"
    echo "== $name $(date +%T)"
    bash "$RUN" "$1" "$2" "$OUT/$name" "$seed" "$3" > "$OUT/$name.out" 2>&1
    grep -E "trainer rc|Traceback|ClientLost|TimeoutError" "$OUT/$name/run.txt"
  done
done
uptime | tee -a "$OUT/load.txt"
echo E49_DONE
