"""Per eval round: how often a reply hits the 16,384-token cap and how trials end (run ON the driver).
    python3 eval_caps.py <run> <label>=<since_epoch>:<until_epoch> ..."""
import json, glob, os, sys, datetime, collections
R = "/mnt/local_storage/charlie/runs/" + sys.argv[1]
tb = set(os.listdir("/mnt/local_storage/charlie/data/tbench21_4cpu"))
ts = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
rounds = [(a.split("=")[0], float(a.split("=")[1].split(":")[0]), float(a.split("=")[1].split(":")[1])) for a in sys.argv[2:]]
rows = collections.defaultdict(list)
for d in glob.glob(R + "/trials/*"):
    if os.path.basename(d).rsplit("__", 1)[0] not in tb: continue
    f = d + "/result.json"
    if not os.path.exists(f): continue
    r = json.load(open(f)); st = ts(r["started_at"])
    lab = next((l for l, a, b in rounds if a <= st < b), None)
    if lab is None: continue
    calls = caps = 0; out = []
    p = d + "/agent/timing.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try: e = json.loads(l)
            except Exception: continue
            if e.get("k") != "llm": continue
            calls += 1
            if e.get("error") == "OutputLengthExceededError": caps += 1
            elif "completion_tokens" in e: out.append(e["completion_tokens"])
    rows[lab].append(((r.get("exception_info") or {}).get("exception_type") or "none",
                      float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0), calls, caps, out))
print("round | trials | mean | calls | capped calls | trials with a capped call (their mean) | out of context | median normal reply tokens | p90")
for lab, _, _ in rounds:
    v = rows[lab]; n = len(v)
    if not n: continue
    calls = sum(x[2] for x in v); caps = sum(x[3] for x in v); wc = [x for x in v if x[3] > 0]
    out = sorted(t for x in v for t in x[4])
    print("%s | %d | %.3f | %d | %d (%.2f%%) | %d (%.3f) | %d (%.1f%%) | %d | %d" % (
        lab, n, sum(x[1] for x in v) / n, calls, caps, 100 * caps / max(1, calls), len(wc), sum(x[1] for x in wc) / max(1, len(wc)),
        sum(x[0] == "ContextLengthExceededError" for x in v), 100 * sum(x[0] == "ContextLengthExceededError" for x in v) / n,
        out[len(out) // 2] if out else 0, out[int(len(out) * .9)] if out else 0))
