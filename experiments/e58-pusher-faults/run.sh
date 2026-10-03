#!/usr/bin/env bash
# E58's runs on guangzhao, W at a time, each worker on its own port.
#   bash run.sh results registered [W]   # PROTOCOL.md; W defaults to 10
#   bash run.sh pilot pilot [W]
#   bash run.sh budget budget [W]
# Writes DIR/<env>/<fault, ':' as '__'>/<dose>/seed<s>/, with train.log in it.
# A traced run is judged by ledger.py as soon as it ends (ledger.json), and
# its trace.npz and ledger.npz are deleted, as in E56. Every run's final policy
# is then evaluated on a clean environment (evaluate.py, clean_eval.json), as in E57.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" OMP_NUM_THREADS=1
PY="$HOME/zuogou/plugrl/bridges-venv/bin/python"
cd "$(dirname "$0")"
DIR="${1:?results dir}"; WHICH="${2:?registered, pilot or budget}"; W="${3:-10}"
mkdir -p "$DIR"
"$PY" jobs.py "$WHICH" > "$DIR/jobs.txt"
uptime | tee "$DIR/load.txt"
(cd "$HOME/zuogou/plugrl/plugrl-bridges" && find src -name "*.py" | sort | xargs sha256sum) > "$DIR/bridges-src.sha256"
sha256sum train.py faulty_client.py jobs.py ledger.py evaluate.py > "$DIR/e58-src.sha256"
echo "$(wc -l < "$DIR/jobs.txt") runs, $W at a time"

worker() {
  local w=$1 port=$((8920 + $1))
  # Every W-th job, starting at the w-th.
  awk -v w="$w" -v W="$W" 'NR % W == w' "$DIR/jobs.txt" | while read -r env fault dose seed trace total; do
    # A process left on the port would take the next run's client (E53).
    if ! "$PY" -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',$port)) else 1)"; then
      echo "worker $w: port $port in use; stopping"; return 1
    fi
    out="$DIR/$env/${fault//:/__}/$dose/seed$seed"
    mkdir -p "$out"
    flag=""; [ "$trace" = trace ] && flag="--trace"
    budget=""; [ -n "${total:-}" ] && budget="--total $total"
    "$PY" train.py --env "$env" --fault "$fault" --dose "$dose" --seed "$seed" \
      --out "$out" --port "$port" $flag $budget > "$out/train.log" 2>&1 || echo "worker $w: train.py exited $? on $out"
    if [ "$trace" = trace ]; then
      if "$PY" ledger.py --run "$out" > "$out/ledger.log" 2>&1; then
        rm -f "$out/trace.npz" "$out/ledger.npz"
      else
        echo "worker $w: ledger.py failed on $out; kept its npz"
      fi
    fi
    "$PY" evaluate.py "$out" "$out" --episodes 50 > "$out/eval.log" 2>&1 || echo "worker $w: evaluate.py failed on $out"
  done
  echo "worker $w done"
}

for w in $(seq 0 $((W - 1))); do
  worker "$w" &
done
wait
uptime | tee -a "$DIR/load.txt"
echo E58_DONE
