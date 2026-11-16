"""Status of a one-pass scan on the Daytona org, against the Dune history of the same tasks.
    python3 scan_status.py <scan run> [history run ...]     (on the driver node)"""
import collections, glob, json, os, statistics, sys, time
from datetime import datetime

D = "/mnt/local_storage/charlie"
run = sys.argv[1]
T = f"{D}/runs/{run}/trials"
TASKS = f"{D}/data/calibforge_v2_daytona"
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
dur = lambda x: (ts(x["finished_at"]) - ts(x["started_at"])) if x and x.get("started_at") and x.get("finished_at") else None
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else float("nan")

hist_cache = f"{D}/ops/calibforge_rl_1110/dune_task_history.json"
if os.path.exists(hist_cache):
    hist = json.load(open(hist_cache))
else:
    acc = collections.defaultdict(lambda: [0, 0.0, 0])
    for h in sys.argv[2:] or ["qwen35-cf-v2-insb-3600-cap16k-t1-1112"]:
        for f in glob.glob(f"{D}/runs/{h}/trials/*/result.json"):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "tbench" in str(r.get("task_id")) or not r.get("task_name"):
                continue
            a = acc[r["task_name"]]
            a[0] += 1
            vr = r.get("verifier_result") or {}
            a[1] += float((vr.get("rewards") or {}).get("reward") or 0.0)
            a[2] += int(bool((r.get("exception_info") or {}).get("exception_type") in ("AgentTimeoutError",)))
    hist = {k: {"n": v[0], "mean": v[1] / v[0], "timeout": v[2] / v[0]} for k, v in acc.items() if v[0] >= 8}
    json.dump(hist, open(hist_cache, "w"))

rows, open_n = [], 0
for d in os.listdir(T):
    try:
        r = json.load(open(f"{T}/{d}/result.json"))
    except Exception:
        open_n += 1
        continue
    e = r.get("exception_info") or {}
    snap = "?"
    try:
        snap = open(f"{TASKS}/{r['task_name']}/environment/cf_snapshot").read().strip().replace("charlieruan-cf-", "")
    except Exception:
        pass
    rows.append({"task": r["task_name"], "exc": e.get("exception_type"), "msg": (e.get("exception_message") or "")[:300],
                 "reward": float(((r.get("verifier_result") or {}).get("rewards") or {}).get("reward") or 0.0),
                 "verified": bool(r.get("verifier_result")), "env": dur(r.get("environment_setup")), "agent_setup": dur(r.get("agent_setup")),
                 "agent": dur(r.get("agent_execution")), "verifier": dur(r.get("verifier")), "snap": snap, "fin": ts(r["finished_at"])})
n = len(rows)
print(f"{time.strftime('%H:%M:%S')} scan {run}: finished {n}, open {open_n}, of {len(os.listdir(TASKS))} tasks")
if not n:
    sys.exit()
print("endings:", dict(collections.Counter(r["exc"] or "none" for r in rows).most_common()))
env = [r["env"] for r in rows if r["env"] is not None and r["agent_setup"] is not None]
print(f"environment setup (create + task setup + stub) s: median {q(env, .5):.0f}, p90 {q(env, .9):.0f}, p99 {q(env, .99):.0f}, max {max(env) if env else 0:.0f} | Dune history ~8 s")
for k in ("agent_setup", "agent", "verifier"):
    v = [r[k] for r in rows if r[k] is not None]
    if v:
        print(f"  {k}: n {len(v)}, median {q(v, .5):.0f} s, p90 {q(v, .9):.0f} s")
fails = [r for r in rows if r["exc"] and r["agent_setup"] is None]
print(f"failed before the agent started: {len(fails)} ({100 * len(fails) / n:.1f}%)", dict(collections.Counter(r["exc"] for r in fails)))
msgs = collections.Counter()
for r in fails:
    m = r["msg"]
    for key in ("CF SETUP FAILED at step", "Too Many Requests", "RBAC", "quota", "No space left", "Unable to locate package", "Temporary failure", "Could not resolve", "Hash Sum mismatch", "429", "timed out", "stub did not start"):
        if key in m:
            msgs[key] += 1
            break
    else:
        msgs[m[:70]] += 1
print("  reasons:", dict(msgs.most_common(8)))
done = [r for r in rows if r["verified"] or r["exc"] in ("AgentTimeoutError", "ContextLengthExceededError")]
print(f"trials with an outcome: {len(done)}, mean reward {statistics.mean(r['reward'] for r in done) if done else 0:.3f}")
cmp_rows = [(r, hist[r["task"]]) for r in done if r["task"] in hist]
if cmp_rows:
    exp = sum(h["mean"] for _, h in cmp_rows)
    obs = sum(r["reward"] for r, _ in cmp_rows)
    var = sum(h["mean"] * (1 - h["mean"]) for _, h in cmp_rows)
    print(f"against Dune history ({len(cmp_rows)} tasks with >= 8 Dune trials): passes observed {obs:.0f}, expected {exp:.0f} +- {var ** .5:.0f} (ratio {obs / exp if exp else 0:.2f})")
    by = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r, h in cmp_rows:
        b = by[r["snap"]]
        b[0] += 1; b[1] += r["reward"]; b[2] += h["mean"]
    for s, b in sorted(by.items()):
        print(f"  {s}: tasks {b[0]}, observed {b[1]:.0f}, expected {b[2]:.0f} (ratio {b[1] / b[2] if b[2] else 0:.2f})")
    sure = [(r, h) for r, h in cmp_rows if h["mean"] >= 0.9]
    lost = [r for r, h in sure if r["reward"] < 0.5]
    print(f"tasks Dune solved >= 90% of the time: {len(sure)}, failed here: {len(lost)} ({100 * len(lost) / max(1, len(sure)):.0f}%) | endings of those: {dict(collections.Counter(r['exc'] or 'wrong answer' for r in lost))}")
    never = [(r, h) for r, h in cmp_rows if h["mean"] <= 0.02]
    print(f"tasks Dune (almost) never solved: {len(never)}, solved here: {sum(1 for r, _ in never if r['reward'] > 0.5)}")
    tmo_obs = sum(1 for r, _ in cmp_rows if r["exc"] == "AgentTimeoutError"); tmo_exp = sum(h["timeout"] for _, h in cmp_rows)
    print(f"agent timeouts: observed {tmo_obs}, expected from Dune {tmo_exp:.0f}")
