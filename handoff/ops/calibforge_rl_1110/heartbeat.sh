#!/usr/bin/env bash
# One-shot heartbeat for the RL run (run ON the driver): status, step times, W&B rows, loop lag, long trials, disk.
D=/mnt/local_storage/charlie; R=${1:-$(cat $D/ops/calibforge_rl_1110/current_run.txt)}; WB=${2:-yjkpuxep}
cd $D; date
bash ops/calibforge_rl_1110/rl_status.sh $R | sed -n 1,3p
bash ops/calibforge_rl_1110/rl_status.sh $R | tail -1
echo "steps done: $(grep -c "Finished: .step., time cost" logs/$R.log); last step times: $(grep -oE "Finished: .step., time cost: [0-9.]+s" logs/$R.log | grep -oE "[0-9]+\." | tr -d . | tail -8 | tr '\n' ' ')"
echo "evals finished: $(grep -oE "Finished: .eval., time cost: [0-9.]+s" logs/$R.log | tr '\n' ' ')"; ls runs/$R/exports/dumped_evals 2>/dev/null | tr '\n' ' '; echo
set -a; . runs/$R/keys.env; set +a; export WANDB_BASE_URL=https://api.wandb.ai
venvs/upstream-megatron/bin/python ops/calibforge_rl_1110/wandb_metrics.py $WB 2>&1 | grep -E "^[0-9]+ " | tail -${3:-5} | cut -c1-330
python3 - "$R" <<'PY'
import json, datetime, glob, os, sys, time, collections
R = "/mnt/local_storage/charlie/runs/" + sys.argv[1]
rows = [json.loads(l) for l in open(R + "/loop_lag.jsonl")]
hr = [r for r in rows if r["t"] > time.time() - 3700]
print("last hour: loop lag max %.1f s | in flight min %d max %d now %d" % (max(r["lag_max"] for r in hr), min(r["trials"] for r in hr), max(r["trials"] for r in hr), rows[-1]["trials"]))
for r in hr:
    if r["lag_max"] > 8: print("  stall", datetime.datetime.fromtimestamp(r["t"]).strftime("%H:%M:%S"), round(r["lag_max"], 1))
ts = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
c = collections.Counter(); rw = []; long = []; now = time.time()
for d in glob.glob(R + "/trials/*"):
    f = d + "/result.json"
    if not os.path.exists(f):
        age = now - os.path.getmtime(d + "/config.json") if os.path.exists(d + "/config.json") else 0
        if age > 4500: long.append((round(age), os.path.basename(d)))
        continue
    if os.path.getmtime(f) < now - 3600: continue
    try: r = json.load(open(f))
    except Exception: continue
    c[(r.get("exception_info") or {}).get("exception_type") or "none"] += 1
    rw.append(float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0))
n = max(1, len(rw))
print("trials finished in the last hour:", len(rw), "mean reward %.3f" % (sum(rw) / n), {k: "%d (%.1f%%)" % (v, 100 * v / n) for k, v in c.most_common(6)})
print("open trials older than 75 min:", sorted(long, reverse=True)[:5])
PY
echo "sampler: $(ps -eo pid,etime,args | grep "pyspy_watch.sh" | grep -v grep | cut -c1-30)"; df -h /mnt/local_storage | tail -1 | awk '{print "disk used", $3, $5}'; ls runs/$R/ckpts 2>/dev/null | tr '\n' ' '; echo
