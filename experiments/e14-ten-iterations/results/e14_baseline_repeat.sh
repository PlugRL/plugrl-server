#!/usr/bin/env bash
# One more cell: the baseline, evaluated a second time.
#
#   setsid nohup bash e14/e14_baseline_repeat.sh > e14/baseline-repeat.out 2>&1 &
#
# PROTOCOL.md registered `eval-iter10-repeat` to test one thing: whether the
# seeding fix makes an evaluation reproducible. It is about to pass without
# testing it. The trained policy scores 0 of 50, so its repeat scores 0 of 50
# too, and a policy that fails every episode agrees with itself under any seed
# or none. The registered prediction will hold vacuously.
#
# The baseline is the only non-degenerate number in the run: 29 of 50. Running
# it a second time with the same seed and the same initial states is the test
# the repeat was meant to be. If it returns 29, the seeding fix does what E12's
# finding said it must; if it returns anything else, that is a real result and
# a more important one than anything else in this experiment.
#
# This cell is additive. It changes no registered value, discards no
# measurement, and replaces nothing - the registered repeat still runs and is
# still reported, vacuous or not.
set -uo pipefail

R=/home/gotham/tmp/plugrl
E=$R/e14
LOG=$E/evals.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "baseline-repeat queued, waiting for the registered cells to finish"
while pgrep -f '[e]14_evals.sh' > /dev/null 2>&1; do sleep 60; done
log "registered cells done, starting baseline-repeat"

bash "$E/e14_eval.sh" eval-baseline-repeat baseline-repeat 8 50 8182 7200 "" \
  > "$E/eval-baseline-repeat.out" 2>&1
log "END eval-baseline-repeat rc=$?"
grep -hE 'client exited with|STAGEC_EVAL_DONE|server never listened' \
  "$E/eval-baseline-repeat.out" 2>/dev/null | tail -3 | tee -a "$LOG"

log "=== rows ==="
cat "$E/results/stageC_eval.tsv" 2>/dev/null | tee -a "$LOG"
log "BASELINE_REPEAT_DONE"
