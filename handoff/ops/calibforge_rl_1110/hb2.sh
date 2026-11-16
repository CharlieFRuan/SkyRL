#!/usr/bin/env bash
# Compact hourly check, run on the head:  hb2.sh <run> <wandb id> <driver ip> [rows=4]
R=$1; WB=$2; IP=$3; N=${4:-4}
date
timeout 115 ssh -p 2222 $IP "cd /mnt/local_storage/charlie; bash ops/calibforge_rl_1110/heartbeat.sh $R $WB $N 2>&1 | grep -E '^eval:|^train:|^steps done|^evals finished|^global_step|^[0-9]+ \{|last hour|stall|trials finished|open trials|tracebacks|sampler|^ *[0-9]+ +[0-9:-]+ bash' | cut -c1-560; python3 ops/calibforge_rl_1110/relay_stats.py $R; echo \"rate-limit lines in log: \$(sed 's/\x1b\[[0-9;]*m//g' logs/$R.log | grep -cE 'DaytonaRateLimit|rate limited \(|Too Many Requests')\"; sed 's/\x1b\[[0-9;]*m//g' logs/$R.log | grep -oE \"(Started|Finished): .(wait_for_generation_buffer|policy_train|eval|save_checkpoints).[^;]{0,24}\" | tail -5 | tr '\n' ';'; echo; ls runs/$R/ckpts 2>/dev/null | tr '\n' ' '; ls runs/$R/exports/dumped_evals 2>/dev/null | tr '\n' ' '; echo"
timeout 60 ssh -p 2222 $IP "cd /mnt/local_storage/charlie; echo \"outage guard lines: \$(sed 's/\x1b\[[0-9;]*m//g' logs/$R.log | grep -c 'outage guard:') | last: \$(sed 's/\x1b\[[0-9;]*m//g' logs/$R.log | grep 'outage guard:' | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}.{0,12}outage guard: .{0,110}' | cut -c1-8,22-140)\"; set -a; . ./${HB_KEYS:-daytona_keys.env}; set +a; timeout 45 venvs/upstream-megatron/bin/python ops/calibforge_rl_1110/key_census.py 2>&1 | tail -1"
echo "backup: $(grep -E 'done global_step|VERIFIED' ckpt_backup.log 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-140)"
echo "watcher: $(ps -eo pid,args | grep '[c]kpt_backup_watch' | cut -c1-110)"
