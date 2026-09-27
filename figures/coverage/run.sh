#!/usr/bin/env bash
# Record and encode every cell in cells.json, one after another.
#
#   bash run.sh [CELL ...]      # all cells, or only those named
#
# record.py runs under the server's venv, pick.py under the env client's;
# both are found under PLUGRL_HOME (default ~/zuogou/plugrl). Each cell's
# server exits after its episodes, so the port is reused.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="${PLUGRL_HOME:-$HOME/zuogou/plugrl}"
SPY="$HOME_DIR/plugrl-server/.venv/bin/python"
CPY="$HOME_DIR/plugrl-env-client/.venv/bin/python"
cd "$HERE"

cells=("$@")
if [ ${#cells[@]} -eq 0 ]; then
  read -r -a cells <<< "$("$SPY" -c 'import json; print(" ".join(c["id"] for c in json.load(open("cells.json"))["cells"]))')"
fi

date '+start %F %T'
failed=0
for cell in "${cells[@]}"; do
  if "$SPY" record.py "$cell" --episodes 5 --port 9700 && "$CPY" pick.py "$cell"; then
    echo "DONE $cell"
  else
    echo "FAILED $cell"
    failed=$((failed + 1))
  fi
done
date '+end   %F %T'
echo "failed: $failed"
echo ALL_DONE
