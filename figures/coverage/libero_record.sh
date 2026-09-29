#!/usr/bin/env bash
# On qz103: one recorded episode of LIBERO-10 task 8, initial state 0, for
# the released pi0.5, FPO and DPPO - three evaluations at once on card pairs
# 0/1, 2/3, 4/5. FPO's is E42's lower-scoring seed (fpopp-s8) at iteration
# 10; it replaced E26's one-iteration control on 2026-09-29, recorded alone
# on cards 0/1. DPPO's is E25's two iterations.
#
#   bash libero_derive.sh && bash libero_record.sh
#
# Initial state 0 is the first state the released policy solves in E26's
# fifty-episode evaluation (its episode 0 succeeds). Each result lands in
# $R/coverage/runs-cov-pi0-*/, and the one-episode scores in
# $R/coverage/results/stageC_eval.tsv. Copy each run's `runs/cov-pi0-X` to
# media/raw/pi0-X/runs/pi0-X on the machine that runs pick.py.
R=/home/gotham/tmp/plugrl
C=$R/coverage
cd "$C"
ALLOW_SIBLINGS=1 EVAL_SRV_GPU=0 EVAL_CLI_GPU=1 setsid nohup bash eval_record.sh cov-pi0-base base 8 1 8701 3600 \
  > eval-base.out 2>&1 < /dev/null &
sleep 20
ALLOW_SIBLINGS=1 EVAL_SRV_GPU=2 EVAL_CLI_GPU=3 setsid nohup bash eval_record.sh cov-pi0-fpo-e42 fpo 8 1 8702 3600 \
  "$R/e42/fpopp-s8/ck/fpo/pi0-policy/fpopp-s8/40960" > eval-fpo-e42.out 2>&1 < /dev/null &
sleep 20
ALLOW_SIBLINGS=1 EVAL_SRV_GPU=4 EVAL_CLI_GPU=5 setsid nohup bash eval_record.sh cov-pi0-dppo dppo 8 1 8703 3600 \
  "$R/e25/runs/dppo-libero/ck/dppo/pi0-policy/dppo-libero/8200" > eval-dppo.out 2>&1 < /dev/null &
echo launched
