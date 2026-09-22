#!/usr/bin/env bash
# Sample the machine's CPU load on a fixed interval, independent of iteration
# boundaries, for the whole of E14.
#
#   setsid nohup bash e14/load_rec.sh > /dev/null 2>&1 &
#
# Why a fixed interval and not one sample per iteration: attribution needs to
# know whether a neighbour's load arrived in the MIDDLE of an iteration.
# Endpoint samples cannot see that, and "we think the neighbour did it" without
# a timeline under it is exactly the unfalsifiable move this project avoids.
#
# One line every 30 s:
#
#   epoch  load1 load5 load15  srv_cpu  cli_cpu cli_n  gpu_apps  top5
#
# - load* comes from /proc/loadavg, which is already a time average and needs
#   no sampling of its own.
# - srv_cpu and cli_cpu are instantaneous, from top's SECOND frame. `ps -o
#   pcpu` would have been cheaper and wrong: it reports an average over the
#   process's whole lifetime, so a neighbour arriving late would be smeared
#   backwards over hours.
# - gpu_apps is pid:mib pairs from nvidia-smi, which identifies exactly who
#   else is on the cards at that moment.
# - top5 is pid:comm:cpu for the five busiest processes, ours or not.
set -uo pipefail

R=/home/gotham/tmp/plugrl
OUT="${1:-$R/e14/load.tsv}"
INTERVAL="${2:-30}"
SRV_PAT='[p]lugrl_server.cli'
CLI_PAT='[v]env-libero.*/bin/python'

[ -s "$OUT" ] || printf 'epoch\tload1\tload5\tload15\tsrv_cpu\tcli_cpu\tcli_n\tgpu_apps\ttop5\n' > "$OUT"

# top's first frame reports CPU since boot; only the second is a real sample.
second_frame() {
  top -bn2 -d 1 2>/dev/null | awk '/^ *PID/ {f++; next} f == 2 && NF >= 12 {print}'
}

while true; do
  read -r l1 l5 l15 _ < /proc/loadavg

  frame=$(second_frame)

  srv_pids=$(pgrep -d'|' -f "$SRV_PAT" 2>/dev/null)
  cli_pids=$(pgrep -d'|' -f "$CLI_PAT" 2>/dev/null)
  cli_n=$(pgrep -cf "$CLI_PAT" 2>/dev/null || echo 0)

  sum_for() {   # pid1|pid2|... -> summed %CPU from the frame
    local pat="$1"
    [ -z "$pat" ] && { echo 0; return; }
    printf '%s\n' "$frame" | awk -v pat="^($pat)$" '$1 ~ pat {s += $9} END {printf "%.1f", s+0}'
  }

  srv_cpu=$(sum_for "$srv_pids")
  cli_cpu=$(sum_for "$cli_pids")

  gpu_apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '{printf "%s:%s|", $1, $2}')
  [ -z "$gpu_apps" ] && gpu_apps="-"

  top5=$(printf '%s\n' "$frame" | sort -k9 -nr | head -5 \
    | awk '{printf "%s:%s:%s|", $1, $12, $9}')
  [ -z "$top5" ] && top5="-"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%s)" "$l1" "$l5" "$l15" "$srv_cpu" "$cli_cpu" "$cli_n" "$gpu_apps" "$top5" >> "$OUT"

  sleep "$INTERVAL"
done
