#!/usr/bin/env bash
# E46's boundary cost, v1 against v2: E43's ladder with an arm in each cell.
#
#   lo    client on guangzhao -> server on guangzhao, 127.0.0.1
#   ts    client on guangzhao -> server on guangzhao, its own Tailscale address
#   phys  client on the laptop -> server on guangzhao, over Tailscale
#
#   v1    bench_client.py as E43 ran it: the observation in every infer
#   v2    bench_client.py --reuse: the infer reuses the feedback's observation
#
# The server is plugrl-protocol's conformance checker in passive mode, as in
# E43, offering reuse-feedback-obs in both arms, so the arms differ only in
# the client. Run from the laptop:
#
#   bash ladder.sh REPS STEPS OUT_TSV
set -uo pipefail

REPS="$1"
STEPS="$2"
TSV="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE="${REMOTE:-guangzhao-memeseeks}"
TS_IP="${TS_IP:-100.75.226.89}"
LOCAL_PY="${LOCAL_PY:?set LOCAL_PY to a Python with msgpack and websockets}"
R_PY='~/zuogou/plugrl/plugrl-server/.venv/bin/python'
R_DIR='~/zuogou/plugrl/e46'
R_BENCH="$R_DIR/bench/bench_client.py"

declare -A PAYLOADS=( [0]="0 1" [48]="128 1" [184]="224 2" [588]="448 1" )
ORDER=(0 48 184 588)

printf 'rung\tarm\tpayload_kib\trep\texchanges\tpack_ms\trtt_ms\tunpack_ms\twall_s\texchanges_per_s\tinfer_bytes\tfeedback_bytes\n' > "$TSV"
date '+start %F %T'

start_server() {  # $1 = port; returns once it listens
  ssh "$REMOTE" "cd $R_DIR && (PYTHONPATH=protocol-src nohup $R_PY -m plugrl_protocol.conformance --host 0.0.0.0 --port $1 --steps $STEPS --timeout 1800 --features reuse-feedback-obs > /tmp/e46srv_$1.log 2>&1 &) ; for i in \$(seq 100); do grep -q 'conformance server on' /tmp/e46srv_$1.log 2>/dev/null && exit 0; sleep 0.1; done; exit 1"
}
server_verdict() {  # $1 = port; the checker's verdict, once it has exited
  # [c], so that the pattern does not match this shell's own command line.
  ssh "$REMOTE" "for i in \$(seq 100); do pgrep -f '[c]onformance --host 0.0.0.0 --port $1 ' >/dev/null || break; sleep 0.1; done; pkill -f '[c]onformance --host 0.0.0.0 --port $1 ' ; grep -E 'violation' /tmp/e46srv_$1.log | tail -1"
}

port=9900
for rep in $(seq "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then RUNGS=(lo ts phys); ARMS=(v1 v2); else RUNGS=(phys ts lo); ARMS=(v2 v1); fi
  for kib in "${ORDER[@]}"; do
    read -r img cams <<< "${PAYLOADS[$kib]}"
    for rung in "${RUNGS[@]}"; do
      for arm in "${ARMS[@]}"; do
        args="--port $port --steps $STEPS --img $img --cameras $cams"
        [ "$arm" = v2 ] && args="$args --reuse"
        start_server "$port" || { echo "server on $port did not start" >&2; port=$((port + 1)); continue; }
        case "$rung" in
          lo)   out=$(ssh "$REMOTE" "$R_PY $R_BENCH --host 127.0.0.1 $args") ;;
          ts)   out=$(ssh "$REMOTE" "$R_PY $R_BENCH --host $TS_IP $args") ;;
          phys) out=$("$LOCAL_PY" "$HERE/bench_client.py" --host "$TS_IP" $args) ;;
        esac
        rc=$?
        verdict=$(server_verdict "$port")
        if [ "$rc" -eq 0 ] && [ -n "$out" ]; then
          printf '%s\t%s\t%s\t%s\t%s\n' "$rung" "$arm" "$kib" "$rep" "$out" >> "$TSV"
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$rung" "$arm" "$kib" "$rep" "$out" "$verdict"
        else
          echo "FAILED $rung $arm $kib rep $rep rc=$rc" >&2
        fi
        port=$((port + 1))
        [ "$port" -ge 9950 ] && port=9900
      done
    done
  done
done
date '+end   %F %T'
echo LADDER_DONE
