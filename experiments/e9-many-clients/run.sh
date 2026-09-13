#!/usr/bin/env bash
# E9 sweep, per experiments/e9-many-clients/PROTOCOL.md.
#
# Only the same-machine column. The second-machine column needs two hosts that
# can reach each other; this node is a container that sees only itself, so that
# column is recorded as not measured, which the protocol's stopping rule
# requires be stated rather than presented as a design choice.
#
# Total environment steps are held fixed across the sweep, so a run with eight
# clients does one eighth of the steps each. Throughput is per server.
set -uo pipefail

ROOT=/home/gotham/tmp/plugrl
export XDG_CACHE_HOME="$ROOT/.cache" TMPDIR="$ROOT/.tmp"
SRV="$ROOT/plugrl-server"
CLI="$ROOT/plugrl-env-client"
SPY="$ROOT/venv/bin/python"
CPY="$ROOT/venv-client/bin/python"

OUT="$ROOT/results-e9"
mkdir -p "$OUT"
TSV="$OUT/summary.tsv"
if [ ! -f "$TSV" ]; then
  printf 'placement\tclients\trep\ttotal_steps\twall_s\tsteps_per_s\tinfer_wait_mean_s\tinfer_wait_p95_s\tno_step_state_warnings\treconnects\tserver_step\tclient_step_sum\tstep_accounting_ok\tvalid\n' > "$TSV"
fi

STEPS="${STEPS:-16384}"
BUFFER="${BUFFER:-4096}"

free_port() {
  local p="$1"
  for pid in $(ss -tlnp 2>/dev/null | awk -v pat=":$p" '$4 ~ pat {print $NF}' | grep -oP 'pid=\K[0-9]+' | sort -u); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 1
}

run_cell() {
  local n="$1" rep="$2" port="$3"
  local tag="same-$n-$rep"
  local slog="$OUT/$tag-server.log" clog="$OUT/$tag-client.log"
  echo "-- clients=$n rep=$rep port=$port"
  # A cell that died early leaves its server listening, and the next run then
  # fails to bind. Take the port back before starting, not only after.
  free_port "$port"

  (cd "$SRV" && "$SPY" -m plugrl_server.cli fpo-policy default fpo default \
     --port "$port" --policy.device cpu \
     --algo.global-steps "$STEPS" --algo.buffer-size "$BUFFER" \
     --no-show-progress-bar --no-show-metric-table \
     --checkpoint-base-dir "$OUT/ck-$tag" --exp-name "$tag" --overwrite \
     > "$slog" 2>&1) &
  local spid=$!
  for _ in $(seq 300); do grep -q "is listening" "$slog" 2>/dev/null && break; sleep 1; done
  if ! grep -q "is listening" "$slog" 2>/dev/null; then
    echo "   server never listened after 300s"; kill $spid 2>/dev/null; return 1
  fi

  local t0=$(date +%s)
  (cd "$CLI" && timeout 1800 "$CPY" -m plugrl_env_client.cli mujoco-v1 \
     --server-host 127.0.0.1 --server-port "$port" \
     --num-envs 1 --num-procs "$n" --num-episodes 100000 \
     --runner.replan-steps 1 --runner.seed 0 > "$clog" 2>&1)
  local cexit=$?
  local wall=$(( $(date +%s) - t0 ))
  # The server stops itself when global-steps is reached. If the client died
  # first it never will, and waiting for it hung one sweep for 21 hours. Give
  # it 60 s to exit on its own, then take it down.
  for _ in $(seq 60); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
  kill -9 "$spid" 2>/dev/null
  free_port "$port"

  # Accounting. Each client process logs one "Final rollout timing summary".
  local csum=$(grep -o "Final rollout timing summary: env_steps=[0-9]*" "$clog" | grep -o "[0-9]*$" | awk '{t+=$1} END{print t+0}')
  csum=${csum:-0}
  local sstep=$(grep -o "Checkpoint saved at step [0-9]*" "$slog" | grep -o "[0-9]*$" | tail -1)
  sstep=${sstep:-0}
  local warn=$(grep -c "arrived with no step state" "$slog")
  local nfinal=$(grep -c "Final rollout timing summary" "$clog")
  local recon=$(( $(grep -c "Waiting for server" "$clog") - n ))
  [ "$recon" -lt 0 ] && recon=0

  local waits=$(grep -o "Final rollout timing summary: .*infer_wait=[0-9.]*" "$clog" | grep -o "infer_wait=[0-9.]*" | cut -d= -f2)
  local wmean=$(echo "$waits" | awk '{s+=$1; n++} END{if(n)printf "%.2f", s/n; else print 0}')
  local wmax=$(echo "$waits" | sort -g | tail -1)
  wmax=${wmax:-0}

  local ok=false
  [ "$sstep" -gt 0 ] && [ "$csum" -ge "$sstep" ] && ok=true
  local valid=false
  [ "$cexit" -eq 0 ] && [ "$nfinal" -eq "$n" ] && valid=true

  local sps=$(awk -v s="$csum" -v w="$wall" 'BEGIN{if(w>0)printf "%.1f", s/w; else print 0}')
  printf 'same\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$n" "$rep" "$csum" "$wall" "$sps" "$wmean" "$wmax" "$warn" "$recon" "$sstep" "$csum" "$ok" "$valid" >> "$TSV"
  echo "   steps=$csum wall=${wall}s sps=$sps warn=$warn recon=$recon server_step=$sstep accounting=$ok valid=$valid"
  rm -rf "$OUT/ck-$tag"
}

port=8900
for n in 1 2 4 8; do
  for rep in 0 1 2; do
    run_cell "$n" "$rep" "$port" || echo "   cell failed"
    port=$((port+1))
  done
done

echo
echo "=== summary ==="
cat "$TSV"
