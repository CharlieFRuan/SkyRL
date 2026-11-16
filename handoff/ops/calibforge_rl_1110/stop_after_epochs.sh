#!/usr/bin/env bash
# Runs ON THE HEAD: stop a fully async run when its N-th epoch has ended and the end-of-epoch checkpoint is saved,
# then upload that checkpoint and delete the run's own sandboxes.
#   setsid nohup bash stop_after_epochs.sh <run> <driver ip> <driver python pid> <epochs to run> [keys file on driver] >> stop_after_epochs.log 2>&1 &
# How an epoch ends (skyrl fully_async_trainer, sample_full_batch): the tasks run out before the nominal step count,
# because groups dropped by the zero-variance filter consume tasks too; the trainer logs
#   "sample_full_batch: epoch <i> exhausted with a partial mini-batch ...; discarding and ending the epoch."
# (i counts from 0), saves a checkpoint named global_step_<last trained step + 1>, and starts the next epoch.
# This script waits for that line with i = <epochs> - 1 and the "Successfully saved checkpoint" line after it.
RUN=$1; DRIVER=$2; PID=$3; EPOCHS=$4; KEYS=${5:-/mnt/local_storage/charlie/daytona_keys.env}
D=/mnt/local_storage/charlie; HERE=/home/ray/hackskyrl/ops/calibforge_rl_1110
SSH="ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=15"
LAST=$((EPOCHS - 1))
say() { echo "$(date '+%F %T') $*"; }
say "watching $RUN (pid $PID on $DRIVER): stop after epoch index $LAST ends"
while true; do
  out=$($SSH $DRIVER "kill -0 $PID 2>/dev/null && echo alive || echo dead; sed 's/\x1b\[[0-9;]*m//g' $D/logs/$RUN.log | awk -v pat='epoch $LAST exhausted with a partial mini-batch' 'index(\$0, pat) {seen=1} seen && /Successfully saved checkpoint for global_step_/ {match(\$0, /global_step_[0-9]+/); print substr(\$0, RSTART+12, RLENGTH-12); exit}'" 2>/dev/null)
  state=$(echo "$out" | head -1); step=$(echo "$out" | sed -n 2p | tr -dc 0-9)
  if [[ "$state" == dead ]]; then say "driver $PID is gone; nothing to stop"; exit 0; fi
  if [[ -n "$step" ]]; then
    say "epoch index $LAST ended; checkpoint global_step_$step saved; SIGTERM to $PID"
    $SSH $DRIVER "kill -TERM $PID; for i in \$(seq 1 18); do sleep 5; kill -0 $PID 2>/dev/null || break; done; if kill -0 $PID 2>/dev/null; then kill -KILL $PID; echo SIGKILL; else echo exited; fi"
    say "uploading global_step_$step"
    if python3 /home/ray/hackskyrl/ops/upload_ckpt_s3.py "$RUN" "$step"; then
      mkdir -p $HERE/ckpt_backup/$RUN; date '+%F %T' > $HERE/ckpt_backup/$RUN/$step.done; say "uploaded global_step_$step"
    else
      say "UPLOAD FAILED for global_step_$step (do it by hand)"
    fi
    say "deleting the run's own sandboxes (owner + run label)"
    $SSH $DRIVER "cd $D && source venvs/upstream-megatron/bin/activate && set -a && . $KEYS && set +a && python ops/evals_1104/dune_cleanup.py $RUN 2>&1 | tail -3"
    say "done"; exit 0
  fi
  sleep 30
done
