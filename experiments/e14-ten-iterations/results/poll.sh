#!/usr/bin/env bash
# One poll of the E14 run. Prints lines a watcher should act on; the watcher
# suppresses lines it has already reported, so a steady state prints nothing.
#
# Progress cannot be read from the runner log - it prints nothing between
# "server listening" and the clients exiting, hours later. Checkpoints landing
# on disk are the progress signal.
#
# The peak memory rides on the progress line rather than a line of its own, so
# it is reported once per iteration instead of every poll: prediction 1 turns
# on whether the allocator grows across ten iterations, and one reading per
# checkpoint is exactly the series that answers it.
R=/home/gotham/tmp/plugrl
E=$R/e14
OUT=$E/ten-iter
CK=$OUT/ck/fpo/pi0-policy/ten-iter

n=$(ls -1 "$CK" 2>/dev/null | grep -cE '^[0-9]+$')
last=$(ls -1 "$CK" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
peak=$(awk -F, '{if ($2>m0) m0=$2; if ($3>m1) m1=$3} END {print m0 "/" m1}' \
  "$OUT/gpu_mem.csv" 2>/dev/null)
echo "progress: ${n}/10 checkpoints, last step ${last:-none}, peak MiB ${peak:-?}"

grep -hoE 'OutOfMemoryError|CUDA out of memory|Killed' "$OUT/server.log" 2>/dev/null \
  | sort -u | sed 's/^/TRAINING SERVER: /'
grep -hE 'clients exit|server never listened|TEARDOWN_NOT_CLEAN' "$E/ten-iter.out" 2>/dev/null
grep -hE 'machine quiet|our own processes|cards free|using gpus|START |END |RUN_REST_|TRAINING_INCOMPLETE|client exited with' \
  "$E/run-rest.log" 2>/dev/null
grep -hE 'EVAL_QUEUE_DONE|EVAL_QUEUE_ABORTED' "$E/eval-queue.log" 2>/dev/null

[ "$(pgrep -cf '[e]14_run_rest.sh')" = 0 ] && echo "ALERT: orchestrator process gone"
exit 0
