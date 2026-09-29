#!/usr/bin/env bash
# E43's boundary cost between two physical machines: E7's ladder with the
# server fixed on guangzhao and the client moved.
#
#   lo    client on guangzhao -> server on guangzhao, 127.0.0.1
#   ts    client on guangzhao -> server on guangzhao, its own Tailscale address
#   phys  client on the laptop -> server on guangzhao, over Tailscale
#
# The server is plugrl-protocol's conformance server, as in E7; the client is
# bench_client.py on both machines. Run from the laptop:
#
#   bash ladder.sh REPS STEPS OUT_TSV
set -uo pipefail

REPS="$1"
STEPS="$2"
TSV="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${REMOTE:-guangzhao-memeseeks}"
TS_IP="${TS_IP:-100.75.226.89}"
LOCAL_PY="${PLUGRL_CLIENT_DIR:?set PLUGRL_CLIENT_DIR}/.venv/Scripts/python.exe"
R_PY='~/zuogou/plugrl/plugrl-server/.venv/bin/python'
R_PROTO='~/zuogou/plugrl/e43-protocol'
R_BENCH='~/zuogou/plugrl/e43-reg/experiments/e43-cross-machine-training/bench_client.py'

declare -A PAYLOADS=( [0]="0 1" [48]="128 1" [184]="224 2" [588]="448 1" )
ORDER=(0 48 184 588)

printf 'rung\tpayload_kib\trep\texchanges\tpack_ms\trtt_ms\tunpack_ms\twall_s\texchanges_per_s\n' > "$TSV"
date '+start %F %T'

start_server() {  # $1 = port; returns once it listens
  ssh "$REMOTE" "cd $R_PROTO && (nohup $R_PY examples/conformance_server.py --host 0.0.0.0 --port $1 --steps $STEPS --timeout 300 > /tmp/e43srv_$1.log 2>&1 &) ; for i in \$(seq 100); do grep -q 'conformance server on' /tmp/e43srv_$1.log 2>/dev/null && exit 0; sleep 0.1; done; exit 1"
}
stop_server() {  # $1 = port
  ssh "$REMOTE" "pkill -f 'conformance_server.py --host 0.0.0.0 --port $1 ' ; true"
}

port=9800
for rep in $(seq "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then RUNGS=(lo ts phys); else RUNGS=(phys ts lo); fi
  for kib in "${ORDER[@]}"; do
    read -r img cams <<< "${PAYLOADS[$kib]}"
    args="--port $port --steps $STEPS --img $img --cameras $cams"
    for rung in "${RUNGS[@]}"; do
      start_server "$port" || { echo "server on $port did not start" >&2; continue; }
      case "$rung" in
        lo)   out=$(ssh "$REMOTE" "$R_PY $R_BENCH --host 127.0.0.1 $args") ;;
        ts)   out=$(ssh "$REMOTE" "$R_PY $R_BENCH --host $TS_IP $args") ;;
        phys) out=$("$LOCAL_PY" "$HERE/bench_client.py" --host "$TS_IP" $args) ;;
      esac
      rc=$?
      stop_server "$port"
      if [ "$rc" -eq 0 ] && [ -n "$out" ]; then
        printf '%s\t%s\t%s\t%s\n' "$rung" "$kib" "$rep" "$out" >> "$TSV"
        printf '%s\t%s\t%s\t%s\n' "$rung" "$kib" "$rep" "$out"
      else
        echo "FAILED $rung $kib rep $rep rc=$rc" >&2
      fi
      port=$((port + 1))
      [ "$port" -ge 9850 ] && port=9800
      args="--port $port --steps $STEPS --img $img --cameras $cams"
    done
  done
done
date '+end   %F %T'
echo LADDER_DONE
