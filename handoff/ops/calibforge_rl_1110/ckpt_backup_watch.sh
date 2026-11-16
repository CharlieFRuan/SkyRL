#!/usr/bin/env bash
# Runs ON THE HEAD (only its role can write the artifact bucket): every 5 min, back up each finished checkpoint of
# the run to S3 with ops/upload_ckpt_s3.py, so a terminated trainer node (11-12 02:45) no longer loses the weights.
#   setsid nohup bash ckpt_backup_watch.sh <run> <driver_ip> >> ckpt_backup.log 2>&1 &
# A checkpoint is "finished" once latest_ckpt_global_step.txt names it (the trainer writes that last).
# Optional 3rd / 4th arguments: upload only steps that are a multiple of UPLOAD_EVERY (default 25), plus any
# checkpoint that is not a multiple of LOCAL_EVERY (default 5: the trainer saves every 5 steps and at each epoch's
# end, so an odd step is an epoch end). The other local checkpoints stay on the trainer node only (user, 11-14:
# "save checkpoints every 5 steps, but only upload the 25 step ones").
UPLOAD_EVERY=${3:-25}; LOCAL_EVERY=${4:-5}
RUN=$1; DRIVER=$2; HERE=/home/ray/hackskyrl/ops/calibforge_rl_1110; MARK=$HERE/ckpt_backup/$RUN; mkdir -p $MARK
SSH="ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=15"
while true; do
  latest=$($SSH $DRIVER "cat /mnt/local_storage/charlie/runs/$RUN/ckpts/latest_ckpt_global_step.txt 2>/dev/null" 2>/dev/null | tr -dc 0-9)
  if [[ -n "$latest" ]]; then
    for step in $($SSH $DRIVER "ls /mnt/local_storage/charlie/runs/$RUN/ckpts 2>/dev/null" | sed -n 's/^global_step_//p' | sort -n); do
      [[ "$step" -le "$latest" && ! -f $MARK/$step.done ]] || continue
      (( step % UPLOAD_EVERY == 0 || step % LOCAL_EVERY != 0 )) || continue
      echo "$(date '+%F %T') uploading $RUN global_step_$step"
      if python3 /home/ray/hackskyrl/ops/upload_ckpt_s3.py "$RUN" "$step"; then
        date '+%F %T' > $MARK/$step.done; echo "$(date '+%F %T') done global_step_$step"
      else
        echo "$(date '+%F %T') FAILED global_step_$step (will retry)"
      fi
    done
  fi
  sleep 300
done
