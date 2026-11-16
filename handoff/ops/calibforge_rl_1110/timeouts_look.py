import json, sys, glob, os
R = sys.argv[1]
base = f"/mnt/local_storage/charlie/runs/{R}/trials"
rows = []
for d in sorted(glob.glob(base + "/*/")):
    rj = d + "result.json"
    if not os.path.exists(rj):
        continue
    try:
        r = json.load(open(rj))
    except Exception:
        continue
    e = r.get("exception_info") or {}
    if e.get("exception_type") != "AgentTimeoutError":
        continue
    tj = d + "agent/timing.jsonl"
    llm, n, keys = [], 0, set()
    if os.path.exists(tj):
        for line in open(tj):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("k") == "llm":
                n += 1
                keys |= set(ev)
                v = ev.get("dt") or 0
                llm.append(float(v))
    rows.append((round(max(llm)) if llm else None, n, round(sum(llm)), r.get("task_name"), sorted(keys)[:8]))
rows.sort(key=lambda x: -(x[0] or 0))
print("timeouts", len(rows), "| longest model call > 900 s:", sum((x[0] or 0) > 900 for x in rows), "| > 300 s:", sum((x[0] or 0) > 300 for x in rows), "| no timing:", sum(x[0] is None for x in rows))
for x in rows[:25]:
    print(x[:4])
print(rows[0][4] if rows else "")
