#!/usr/bin/env bash
# Every 15 s: a non-blocking py-spy dump of the generator process (the skyrl_entrypoint actor), so a stall of its event
# loop (loop_lag.jsonl) can be matched to a stack afterwards.   pyspy_watch.sh <pid> <out dir>
PID=$1; OUT=$2; PS=/mnt/local_storage/charlie/tools/pyspy/bin/py-spy
while kill -0 $PID 2>/dev/null; do
  timeout 10 $PS dump --nonblocking --pid $PID 2>&1 | gzip > $OUT/$(date +%m%d-%H%M%S).txt.gz
  sleep 15
done
