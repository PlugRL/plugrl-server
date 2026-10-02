#!/usr/bin/env bash
# E48 cross arm: do what cross_clients.py's poll was meant to do. Its
# --run-dir (/home/guangzhao/...) reached it as "D:/Program Files/Git/home/..."
# - Git Bash converts POSIX paths passed to Windows programs - so the remote
# grep never found RUN_DONE, and the laptop's clients kept reconnecting after
# RLinf had finished. For each cross seed: wait for RUN_DONE on guangzhao,
# then stop that seed's client tree here, which lets cross_clients.py and
# run_e48.sh carry on. Logs what it did, with times, for AMENDMENT.md.
set -uo pipefail
LOG=/d/75128/Desktop/plugrl-work/plugrl-server/.claude/worktrees/fix-pi0-image-mask-batching/experiments/e48-rlinf-bridge/results/stop-watcher.txt
for seed in 0 1 2; do
  until ssh -n -o BatchMode=yes -o ConnectTimeout=10 guangzhao-memeseeks \
      "grep -q RUN_DONE ~/zuogou/plugrl/e48/cross-seed$seed/run.txt"; do
    sleep 15
  done
  seen=$(date '+%F %T')
  pid=""
  until [ -n "$pid" ]; do
    pid=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'plugrl-env-client/.venv' -and \$_.CommandLine -match 'rlinf-cross-clients' -and \$_.CommandLine -match '--runner.seed $seed ' }).ProcessId" | tr -d '\r' | head -1)
    [ -n "$pid" ] || sleep 5
  done
  MSYS_NO_PATHCONV=1 taskkill /T /F /PID "$pid" > /dev/null 2>&1
  echo "cross seed $seed: RUN_DONE seen $seen; stopped client tree $pid at $(date '+%T')" | tee -a "$LOG"
done
echo WATCHER_DONE | tee -a "$LOG"
