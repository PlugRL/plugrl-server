#!/usr/bin/env bash
# E54's runs on guangzhao, W at a time, each worker on its own port.
#   bash run.sh results registered [W]   # PROTOCOL.md; W defaults to 10
#   bash run.sh pilot pilot [W]
# Writes DIR/<env>/<fault, ':' as '__'>/<dose>/seed<s>/, with train.log in it.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
PY="$HOME/zuogou/plugrl/bridges-venv/bin/python"
cd "$(dirname "$0")"
DIR="${1:?results dir}"; WHICH="${2:?registered or pilot}"; W="${3:-10}"
mkdir -p "$DIR"
"$PY" jobs.py "$WHICH" > "$DIR/jobs.txt"
uptime | tee "$DIR/load.txt"
(cd "$HOME/zuogou/plugrl/plugrl-bridges" && find src -name "*.py" | sort | xargs sha256sum) > "$DIR/bridges-src.sha256"
sha256sum train.py faulty_client.py jobs.py > "$DIR/e54-src.sha256"
echo "$(wc -l < "$DIR/jobs.txt") runs, $W at a time"

worker() {
  local w=$1 port=$((8854 + $1)) n=0
  # Every W-th job, starting at the w-th.
  awk -v w="$w" -v W="$W" 'NR % W == w' "$DIR/jobs.txt" | while read -r env fault dose seed trace; do
    # A process left on the port would take the next run's client (E53).
    if ! "$PY" -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$port)) else 1)"; then
      echo "worker $w: port $port in use; stopping"; return 1
    fi
    out="$DIR/$env/${fault//:/__}/$dose/seed$seed"
    mkdir -p "$out"
    flag=""; [ "$trace" = trace ] && flag="--trace"
    "$PY" train.py --env "$env" --fault "$fault" --dose "$dose" --seed "$seed" \
      --out "$out" --port "$port" $flag > "$out/train.log" 2>&1 || echo "worker $w: train.py exited $? on $out"
    n=$((n + 1))
  done
  echo "worker $w done"
}

for w in $(seq 0 $((W - 1))); do
  worker "$w" &
done
wait
uptime | tee -a "$DIR/load.txt"
echo E54_DONE
