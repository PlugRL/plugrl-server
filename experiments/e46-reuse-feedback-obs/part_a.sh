#!/usr/bin/env bash
# E46 Part A from the laptop: the path, the throughput, the ladder, the path.
#   bash part_a.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$HERE/results"
mkdir -p "$R"
path() { tailscale status 2>&1 | grep guangzhao; tailscale ping -c 1 100.75.226.89 2>&1 | head -1; }
{ date '+path before %F %T'; path; } | tee "$R/path.txt"
bash "$HERE/throughput.sh" 5 64 "$R/throughput.tsv"
{ date '+path after throughput %F %T'; path; } | tee -a "$R/path.txt"
bash "$HERE/ladder.sh" 5 1000 "$R/ladder.tsv"
{ date '+path after ladder %F %T'; path; } | tee -a "$R/path.txt"
echo PART_A_DONE
