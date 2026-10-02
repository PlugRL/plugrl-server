#!/usr/bin/env bash
# E50's 12 registered runs on guangzhao, one at a time (PROTOCOL.md).
#   bash run.sh            # from ~/zuogou/plugrl/e50, writes results/<env>-<arm>-seed<s>/
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
PY="$HOME/zuogou/plugrl/bridges-venv/bin/python"
cd "$(dirname "$0")"
mkdir -p results
uptime | tee results/load.txt
# plugrl-bridges is not pushed yet: record the source the runs used.
(cd "$HOME/zuogou/plugrl/plugrl-bridges" && find src -name "*.py" | sort | xargs sha256sum) | tee results/bridges-src.sha256
for env in pendulum halfcheetah; do
  for seed in 0 1 2; do
    for arm in inprocess bridge; do
      echo "== $env $arm seed $seed $(date +%T)"
      "$PY" train.py --env "$env" --arm "$arm" --seed "$seed" \
        --out "results/$env-$arm-seed$seed" --port 8820 > "results/$env-$arm-seed$seed.log" 2>&1
      tail -n 1 "results/$env-$arm-seed$seed.log"
    done
  done
done
uptime | tee -a results/load.txt
echo E50_DONE
