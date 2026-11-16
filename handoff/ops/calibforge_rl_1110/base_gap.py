"""Why do two evals of the same model differ? Endings, per-task flips and what binomial noise predicts.
    python3 base_gap.py <run A> <run B> [more runs to pool for the per-task pass rate ...]   (on the driver)"""
import json, os, sys, collections, datetime, math, statistics as st
D = "/mnt/local_storage/charlie"
tb = set(os.listdir(f"{D}/data/tbench21_own"))
ts = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def dur(r, k):
    x = r.get(k) or {}
    try:
        return ts(x["finished_at"]) - ts(x["started_at"])
    except Exception:
        return None


def load(run):
    base = f"{D}/runs/{run}/trials"; by = collections.defaultdict(list)
    for t in os.listdir(base):
        task = t.rsplit("__", 1)[0]
        if task not in tb or not os.path.exists(f"{base}/{t}/result.json"):
            continue
        try:
            r = json.load(open(f"{base}/{t}/result.json"))
        except Exception:
            continue
        et = (r.get("exception_info") or {}).get("exception_type") or "none"
        rw = float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0) or 0)
        by[task].append(dict(st=ts(r["started_at"]), rw=rw, et=et, agent=dur(r, "agent_execution"), env=dur(r, "environment_setup"),
                             ver=dur(r, "verifier"), has_ver=r.get("verifier_result") is not None))
    for v in by.values():
        v.sort(key=lambda x: x["st"])
        del v[:-3]  # a retried attempt replaces the first one
    return by


runs = {r: load(r) for r in sys.argv[1:]}
A, B = sys.argv[1], sys.argv[2]


def kind(x):
    if x["rw"] >= 1:
        return "pass"
    if x["et"] == "none":
        return "fail: agent finished, tests failed"
    return "fail: " + x["et"] + (" (tests ran)" if x["has_ver"] else " (no test result)")


for r in (A, B):
    tr = [x for v in runs[r].values() for x in v]
    c = collections.Counter(kind(x) for x in tr)
    med = lambda k: st.median([x[k] for x in tr if x[k] is not None])
    print(f"{r[-22:]}: trials {len(tr)} passes {c['pass']} | agent s median {med('agent'):.0f} | env {med('env'):.0f} | verifier {med('ver'):.0f}")
    for k, v in c.most_common():
        print(f"     {v:4d}  {k}")
# passes among timed-out trials (a timeout still gets its tests run)
for r in (A, B):
    to = [x for v in runs[r].values() for x in v if x["et"] == "AgentTimeoutError"]
    print(f"{r[-22:]}: agent timeouts {len(to)}, of which passed {sum(x['rw'] >= 1 for x in to)}")
a, b = runs[A], runs[B]
tasks = sorted(set(a) & set(b))
pa = {t: sum(x["rw"] for x in a[t]) for t in tasks}; pb = {t: sum(x["rw"] for x in b[t]) for t in tasks}
print(f"passes A {sum(pa.values()):.0f}, B {sum(pb.values()):.0f}, difference {sum(pa.values()) - sum(pb.values()):+.0f} trials")
lost = collections.Counter(); gained = collections.Counter()
for t in tasks:
    if pa[t] > pb[t]:
        for x in b[t]:
            if x["rw"] < 1: lost[kind(x)] += 1
    if pb[t] > pa[t]:
        for x in a[t]:
            if x["rw"] < 1: gained[kind(x)] += 1
print("tasks where A > B:", sum(pa[t] > pb[t] for t in tasks), "| failing trials of B on them:", dict(lost.most_common()))
print("tasks where B > A:", sum(pb[t] > pa[t] for t in tasks), "| failing trials of A on them:", dict(gained.most_common()))
# binomial expectation with the per-task pass rate pooled over every run given
pool = {t: [x["rw"] for r in runs for x in runs[r].get(t, [])] for t in tasks}
p = {t: sum(v) / len(v) for t, v in pool.items()}
var = sum(3 * p[t] * (1 - p[t]) for t in tasks)
n = 3 * len(tasks)
print(f"pooled over {len(runs)} runs ({len(next(iter(pool.values())))} trials per task): mean pass rate {sum(p.values()) / len(p):.4f}; tasks never solved {sum(v == 0 for v in p.values())}, always {sum(v == 1 for v in p.values())}, in between {sum(0 < v < 1 for v in p.values())}")
print(f"binomial sd of one eval's avg: {math.sqrt(var) / n:.4f}; of the difference of two evals: {math.sqrt(2 * var) / n:.4f}; observed A - B: {(sum(pa.values()) - sum(pb.values())) / n:+.4f}")
print("per-run avg:", {r[-22:]: round(sum(x['rw'] for t in tasks for x in runs[r].get(t, [])) / max(1, sum(len(runs[r].get(t, [])) for t in tasks)), 4) for r in runs})
