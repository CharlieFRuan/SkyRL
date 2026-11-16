#!/usr/bin/env bash
# Status of the CalibForge RL run (run ON the driver node): trials by phase (eval = TB2.1 tasks), errors, steps.
D=/mnt/local_storage/charlie; RUN=${1:-$(cat $D/ops/calibforge_rl_1110/current_run.txt)}
cd $D/runs/$RUN/trials 2>/dev/null || { echo "no trials dir"; exit 0; }
python3 - "$D" <<'PY'
import glob, json, os, sys, collections
D = sys.argv[1]
tb = set(os.listdir(f"{D}/data/tbench21_own"))
s = {"eval": collections.Counter(), "train": collections.Counter()}; rw = {"eval": [], "train": []}
started = collections.Counter()
for t in os.listdir("."):
    ph = "eval" if t.rsplit("__", 1)[0] in tb else "train"
    started[ph] += 1
    f = f"{t}/result.json"
    if not os.path.exists(f): continue
    try: r = json.load(open(f))
    except Exception: continue
    e = (r.get("exception_info") or {}).get("exception_type") or "none"
    s[ph][e] += 1
    rw[ph].append(float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0))
for ph in ("eval", "train"):
    n = len(rw[ph]); print(f"{ph}: started {started[ph]}, done {n}, mean {sum(rw[ph])/max(1,n):.3f}, exc {dict(s[ph].most_common(5))}")
PY
L=$D/logs/$RUN.log
echo "tracebacks: $(grep -c 'Traceback (most recent' $L)"
grep -oE "'(policy|generate|trainer|loss|reward|eval)[^']*avg_score[^']*': [0-9.]+|eval/all/[a-z_@0-9]+: [0-9.]+|global_step[=: ]+[0-9]+" $L | tail -4
tail -c 600 $L | tr '\r' '\n' | grep -E 'Generating|Evaluation|step' | tail -2 | cut -c1-200
grep -oE "Finished: '(step|eval[a-z_]*|wait_for_generation_buffer|policy_train)', time cost: [0-9.]+s" $L | tail -5 | tr '\n' ';'; echo
python3 - "$D/runs/$RUN" <<'PY'
import json, os, sys
p = sys.argv[1] + "/loop_lag.jsonl"
if os.path.exists(p):
    lag = [json.loads(l) for l in open(p)][-30:]
    if lag: print("generator loop, last 5 min: lag mean %.3f s, max %.1f s, sandboxes up %d" % (sum(x["lag_mean"] for x in lag) / len(lag), max(x["lag_max"] for x in lag), lag[-1]["trials"]))
PY
ps -eo pid,etime,args | grep "runs/$RUN/run_fully_async_terminus.sh" | grep -v grep | cut -c1-60
