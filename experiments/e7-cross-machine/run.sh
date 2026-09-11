#!/usr/bin/env bash
# E7: what the process boundary costs once packets leave loopback.
#
# Follows PROTOCOL.md, which was committed before this script ran.
#
# Four rungs, not the two the protocol names. The extra pair exists because
# of a problem the protocol did not anticipate: with only one machine, no
# configuration changes the network path and nothing else. Going from
# loopback to the Windows host moves the server to a different operating
# system and a different Python at the same time. So:
#
#   lo    client in WSL2  ->  server in WSL2, 127.0.0.1     both Linux, loopback
#   self  client in WSL2  ->  server in WSL2, its eth0 IP   both Linux, via the stack
#   win   client in WSL2  ->  server on Windows, host IP    crosses the hypervisor too
#
# `self` minus `lo` is the cost of the network stack with the operating
# system held fixed. `win` minus `self` is what crossing to the host adds.
# Reporting only `win` minus `lo` would attribute the whole gap to the
# network, which would be wrong.
#
# Usage: bash run.sh [REPS] [EXCHANGES]
set -euo pipefail

REPS="${1:-5}"
STEPS="${2:-120}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"
mkdir -p "$OUT"

CLIENT_SRC="$(cd "$HERE/../../../plugrl-protocol/examples" && pwd)"
# Git Bash gives /d/...; WSL wants /mnt/d/... . wslpath is no help here
# because it expects a Windows path, and handed /d/... it produces
# /mnt/d/d/... .
CLIENT_SRC_WSL="$(echo "$CLIENT_SRC" | sed -E 's|^/([a-zA-Z])/|/mnt/\1/|')"
WIN_PY="$(cd "$HERE/../.." && pwd)/../plugrl-env-client/.venv/Scripts/python.exe"

WSL_SELF=$(wsl.exe -e bash -lc "ip -4 addr show eth0 | awk '/inet /{print \$2}' | cut -d/ -f1" | tr -d '\r')
WSL_HOST=$(wsl.exe -e bash -lc "ip route | awk '/^default/ {print \$3}'" | tr -d '\r')

echo "WSL2 eth0    = $WSL_SELF"
echo "Windows host = $WSL_HOST"

# One binary for every measurement, built once, so the client is never a
# variable.
wsl.exe -e bash -lc "cd '$CLIENT_SRC_WSL' && g++ -std=c++17 -O2 -o /tmp/e7_client plugrl_client.cpp" \
  || { echo "client build failed" >&2; exit 1; }

TSV="$OUT/summary.tsv"
printf 'rung\tpayload_kib\trep\texchanges\tpack_ms\trtt_ms\tunpack_ms\twall_s\texchanges_per_s\tper_exchange_per_s\tvalid\n' > "$TSV"

# payload label -> client args "img cameras"
declare -A PAYLOADS=( [48]="128 1" [184]="224 2" [588]="448 1" )

start_server() {  # $1 = where (wsl|win), $2 = port
  if [ "$1" = "wsl" ]; then
    # `wsl.exe -e bash -lc "... &"` does not work: when that command returns,
    # WSL tears the session down and takes the backgrounded child with it,
    # nohup or no nohup - no log file is even created. Backgrounding wsl.exe
    # itself on the Windows side keeps the session alive for as long as the
    # server runs, and `exec` makes the server the process wsl.exe waits on.
    (wsl.exe -e bash -lc "cd '$CLIENT_SRC_WSL' && exec python3 conformance_server.py --host 0.0.0.0 --port $2 --steps 100000 --timeout 300" > "$OUT/srv_wsl_$2.log" 2>&1 &) || true
  else
    (cd "$CLIENT_SRC" && "$WIN_PY" conformance_server.py --host 0.0.0.0 --port "$2" --steps 100000 --timeout 300 > "$OUT/srv_win_$2.log" 2>&1 &) || true
  fi
}

stop_servers() {
  wsl.exe -e bash -lc "pkill -f conformance_server.py" >/dev/null 2>&1 || true
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'conformance_server' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" >/dev/null 2>&1 || true
}
trap stop_servers EXIT

port=8940
for rung in lo self win; do
  case "$rung" in
    lo)   where=wsl; target=127.0.0.1 ;;
    self) where=wsl; target="$WSL_SELF" ;;
    win)  where=win; target="$WSL_HOST" ;;
  esac

  for kib in 48 184 588; do
    args="${PAYLOADS[$kib]}"
    for rep in $(seq 1 "$REPS"); do
      port=$((port + 1))
      stop_servers; sleep 1
      start_server "$where" "$port"

      # Wait for the server to say it is up rather than sleeping blind.
      ready=0
      for _ in $(seq 40); do
        log="$OUT/srv_${where}_$port.log"
        grep -q 'conformance server on' "$log" 2>/dev/null && { ready=1; break; }
        sleep 1
      done
      [ "$ready" = "1" ] || { printf '%s\t%s\t%s\t0\t\t\t\t\t\t\tfalse\n' "$rung" "$kib" "$rep" >> "$TSV"; continue; }

      t0=$(date +%s.%N)
      out=$(wsl.exe -e bash -lc "timeout 200 /tmp/e7_client $target $port $STEPS 1 $args 2>&1" || true)
      t1=$(date +%s.%N)

      # A run counts only if the client says it finished - PROTOCOL.md, and
      # the trap E5's harness fell into.
      if echo "$out" | grep -q "completed $STEPS infer/action/feedback exchanges"; then
        line=$(echo "$out" | grep -oE "TSV[[:space:]].*" | tail -1)
        pack=$(echo "$line" | awk '{print $6}')
        rtt=$(echo "$line" | awk '{print $7}')
        unpack=$(echo "$line" | awk '{print $8}')
        wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
        eps=$(awk -v n="$STEPS" -v w="$wall" 'BEGIN{printf "%.1f", (w>0)? n/w : 0}')
        # Wall-clock throughput includes process startup, which at short run
        # lengths is most of it. This one comes from the client's own
        # per-exchange timings and excludes startup entirely. PROTOCOL.md
        # named the wall-clock figure; FINDINGS.md says why the claim does
        # not rest on it.
        pxs=$(awk -v p="$pack" -v r="$rtt" -v u="$unpack" 'BEGIN{t=p+r+u; printf "%.1f", (t>0)? 1000.0/t : 0}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\ttrue\n' \
          "$rung" "$kib" "$rep" "$STEPS" "$pack" "$rtt" "$unpack" "$wall" "$eps" "$pxs" >> "$TSV"
        echo "  $rung ${kib}KiB rep$rep: rtt=${rtt}ms  ${eps}/s"
      else
        printf '%s\t%s\t%s\t0\t\t\t\t\t\t\tfalse\n' "$rung" "$kib" "$rep" >> "$TSV"
        echo "  $rung ${kib}KiB rep$rep: INVALID (client did not confirm completion)"
      fi
    done
  done
done

stop_servers
echo
echo "wrote $TSV"
