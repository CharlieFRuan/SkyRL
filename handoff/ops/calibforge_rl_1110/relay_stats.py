"""Sum the relay managers' /stats for a run:  relay_stats.py <run>   (run on the driver node)."""
import collections, glob, json, sys, urllib.request

R = "/mnt/local_storage/charlie/runs/" + sys.argv[1]
managers = set()
for f in sorted(glob.glob(R + "/trials/*/config.json"))[-400:]:
    try:
        managers.add(json.load(open(f))["environment"]["kwargs"]["skycap_relay"]["manager"])
    except Exception:
        pass
total, types, n = collections.Counter(), collections.Counter(), 0
for m in sorted(managers):
    try:
        s = json.loads(urllib.request.urlopen(m + "/stats", timeout=5).read())
    except Exception as e:
        print(m, "stats failed", type(e).__name__)
        continue
    n += 1
    for k, v in s.items():
        if isinstance(v, (int, float)):
            total[k] += v
    types.update(s.get("error_types") or {})
out = {k: (round(v / n, 3) if k.startswith("mean") else int(v)) for k, v in total.items()}
print(f"relay: managers {n}/{len(managers)} | " + " ".join(f"{k}={out[k]}" for k in ("polling", "attached", "calls", "delta_calls", "poll_errors", "call_errors", "setup_errors", "dropped", "rate_limited", "recovered", "duplicates", "mean_overhead_seconds") if k in out))
print("relay error types:", dict(types))
