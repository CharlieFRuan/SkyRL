"""Eval summary for one eval round of the RL run (run ON the driver).  python3 eval_step.py <run> [since_epoch] [until_epoch]"""
import json, os, sys, collections, datetime
D = "/mnt/local_storage/charlie"
run = sys.argv[1]
since = float(sys.argv[2]) if len(sys.argv) > 2 else 0
until = float(sys.argv[3]) if len(sys.argv) > 3 else 1e18
tb = set(os.listdir(f"{D}/data/tbench21_own"))
base = f"{D}/runs/{run}/trials"
ts = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
by = collections.defaultdict(list)
for t in os.listdir(base):
    task = t.rsplit("__", 1)[0]
    if task not in tb: continue
    f = f"{base}/{t}/result.json"
    if not os.path.exists(f): continue
    try: r = json.load(open(f))
    except Exception: continue
    st = ts(r["started_at"])
    if not (since <= st < until): continue
    et = (r.get("exception_info") or {}).get("exception_type") or "none"
    rw = float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0)
    run_away = False
    p = f"{base}/{t}/agent/timing.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try: d = json.loads(l)
            except Exception: continue
            if d.get("k") == "llm" and d.get("error") == "OutputLengthExceededError" and d.get("dt", 0) > 120: run_away = True
    by[task].append((st, rw, et, run_away, t))
# A retried attempt replaces the first: keep the last 3 per task (the generator retries a trial that produced nothing).
trials = []
for task, v in by.items():
    v.sort()
    trials += [(task,) + x for x in v]
n = len(trials)
kinds = collections.Counter(("runaway turn" if x[4] else x[3]) for x in trials)
mean = sum(x[2] for x in trials) / max(1, n)
tasks = collections.defaultdict(list)
for x in trials: tasks[x[0]].append(x[2])
print(f"trials {n} tasks {len(tasks)} per-trial mean {mean:.3f}")
print("per-task mean of means %.3f, pass@k (any) %.3f, all-pass %.3f" % (
    sum(sum(v) / len(v) for v in tasks.values()) / len(tasks),
    sum(1 for v in tasks.values() if max(v) >= 1) / len(tasks),
    sum(1 for v in tasks.values() if min(v) >= 1) / len(tasks)))
print("endings:", dict(kinds.most_common()))
ok = [x for x in trials if not x[4]]
print("excluding runaway-turn trials: n %d mean %.3f" % (len(ok), sum(x[2] for x in ok) / max(1, len(ok))))
clean = [x for x in trials if x[3] == "none"]
print("clean trials: n %d mean %.3f" % (len(clean), sum(x[2] for x in clean) / max(1, len(clean))))
print("trials per task:", dict(collections.Counter(len(v) for v in tasks.values())))
