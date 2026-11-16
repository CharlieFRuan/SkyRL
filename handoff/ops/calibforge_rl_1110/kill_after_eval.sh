#!/usr/bin/env bash
# Runs ON THE DRIVER NODE: when a run's eval round has written its results, stop that run's driver (by PID).
#   setsid nohup bash kill_after_eval.sh <run> <driver python pid> <step> > /dev/null 2>&1 &
# For eval-only use of the training entrypoint (eval_before_train, then nothing else should run).
R=$1; PID=$2; STEP=$3; D=/mnt/local_storage/charlie
F=$D/runs/$R/exports/dumped_evals/global_step_${STEP}_evals/aggregated_results.jsonl
while kill -0 $PID 2>/dev/null; do
  if [ -s "$F" ]; then
    sleep 20   # let the trainer hand the eval row to W&B
    kill -TERM $PID; date "+%F %T eval $STEP of $R written; SIGTERM to $PID" >> $D/logs/$R.kill.log
    for i in $(seq 1 12); do sleep 5; kill -0 $PID 2>/dev/null || break; done
    kill -0 $PID 2>/dev/null && { kill -KILL $PID; date "+%F %T SIGKILL to $PID" >> $D/logs/$R.kill.log; }
    exit 0
  fi
  sleep 3
done
