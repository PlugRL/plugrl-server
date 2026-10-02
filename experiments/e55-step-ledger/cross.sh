#!/usr/bin/env bash
# E55's runs across machines, from the laptop: SB3 and the bridge on guangzhao
# (train.py --external-client), the env client here, over Tailscale. One run at
# a time: none, then E54's 21 client faults and 2 log faults at one cell.
#   bash cross.sh results/cross [SEED]       # SEED defaults to 0
#   FAULTS="none obs:stale" bash cross.sh pilot/cross 9   # only these faults
# Each run's files from both machines end up in DIR/pendulum/<fault>/<dose>/seed<s>/.
set -uo pipefail
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
H=guangzhao-memeseeks
R=zuogou/plugrl/exp/e55-step-ledger
GZ=100.75.226.89
PORT=8880
PY=/c/Users/75128/.claude/jobs/1fea619d/tmp/plugrl-rlinf/.venv/Scripts/python.exe
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
DIR="${1:?results dir}"; SEED="${2:-0}"
CLIENT_FAULTS=$("$PY" -c "from faulty_client import SILENT; print(' '.join(SILENT))")
LOG_FAULTS="log:drop-first log:short"
mkdir -p "$DIR"
"$PY" --version > "$DIR/laptop-python.txt" 2>&1
"$PY" -c "import numpy, gymnasium; print('numpy', numpy.__version__, 'gymnasium', gymnasium.__version__)" >> "$DIR/laptop-python.txt"

for fault in ${FAULTS:-none $CLIENT_FAULTS $LOG_FAULTS}; do
  dose=one; [ "$fault" = none ] && dose=1.0
  out="$DIR/pendulum/${fault//:/__}/$dose/seed$SEED"
  mkdir -p "$out"
  client_fault=$fault; case $fault in log:*) client_fault=none ;; esac
  echo "== $fault $dose seed $SEED $(date +%T)"
  ssh -n -o BatchMode=yes "$H" "cd ~/$R && mkdir -p $out && (OMP_NUM_THREADS=1 nohup ~/zuogou/plugrl/bridges-venv/bin/python train.py --env pendulum --fault $fault --dose $dose --seed $SEED --out $out --port $PORT --trace --external-client > $out/train.log 2>&1 &)"
  "$PY" faulty_client.py --host "$GZ" --port "$PORT" --num-envs 16 --seed "$SEED" \
    --env-id Pendulum-v1 --action-high 2.0 --fault "$client_fault" --dose "$dose" \
    --record "$out/env_record.csv" --ledger "$out/ledger.npz" > "$out/client.log" 2>&1
  echo "client exit $?" >> "$out/client.log"
  tail -n 2 "$out/client.log"
  waited=0
  until ssh -n -o BatchMode=yes "$H" "test -f ~/$R/$out/result.json"; do
    sleep 2; waited=$((waited + 2))
    if [ "$waited" -ge 300 ]; then echo "no result.json after 300 s: $out"; break; fi
  done
  ssh -n -o BatchMode=yes "$H" "cd ~/$R/$out && tar -cf - result.json trace.npz monitor.csv episodes.csv train.log" | tar --force-local -xf - -C "$out"
done
echo E55_CROSS_DONE
