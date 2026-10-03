#!/usr/bin/env bash
# E57 on guangzhao: evaluate every E54 and E56 final policy on a clean
# environment, 12 at a time. Writes results/<env>/<fault>/<dose>/seed<s>/clean_eval.json.
#   bash run.sh
set -uo pipefail
export OMP_NUM_THREADS=1
PY="$HOME/zuogou/plugrl/bridges-venv/bin/python"
cd "$(dirname "$0")"
mkdir -p results
find ../e54-boundary-faults/results ../e56-hopper-faults/results -name model.zip \
  -not -path '*/pilot/*' -printf '%h\n' | sort > results/jobs.txt
sha256sum evaluate.py summarise.py > results/e57-src.sha256
uptime | tee results/load.txt
echo "$(wc -l < results/jobs.txt) runs, 12 at a time"
xargs -P 12 -I{} sh -c '
  r="$1"; rel="${r#*/results/}"
  "$0" evaluate.py "$r" "results/$rel" --episodes 50 > /dev/null 2> "results/$(echo "$rel" | tr / _).err" \
    && rm -f "results/$(echo "$rel" | tr / _).err" || echo "failed: $r"
' "$PY" {} < results/jobs.txt
uptime >> results/load.txt
echo E57_DONE
