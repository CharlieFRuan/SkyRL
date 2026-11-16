"""Environment-setup time of trials that started in the last N minutes:  env_setup_recent.py <run> [minutes=75]"""
import json, sys, glob, os, time, statistics as st
from datetime import datetime
R = sys.argv[1]; N = float(sys.argv[2]) if len(sys.argv) > 2 else 75
base = f"/mnt/local_storage/charlie/runs/{R}/trials"
now = time.time(); durs, slow, waiting = [], [], []
for d in glob.glob(base + "/*/"):
    try:
        c = os.path.getmtime(d + "config.json")
    except OSError:
        continue
    if now - c > N * 60:
        continue
    rj = d + "result.json"
    if os.path.exists(rj):
        try:
            r = json.load(open(rj)); e = r.get("environment_setup") or {}
            dt = (datetime.fromisoformat(e["finished_at"]) - datetime.fromisoformat(e["started_at"])).total_seconds()
        except Exception:
            continue
        durs.append(dt)
        if dt > 120:
            slow.append(round(dt))
    elif not os.path.exists(d + "exception.txt") and not os.path.isdir(d + "agent/setup") and not os.listdir(d + "agent") if os.path.isdir(d + "agent") else False:
        waiting.append(round(now - c))
durs.sort()
if durs:
    print(f"finished trials started in the last {N:.0f} min: {len(durs)} | env setup median {st.median(durs):.0f} p90 {durs[int(.9*len(durs))]:.0f} p99 {durs[int(.99*len(durs))]:.0f} max {durs[-1]:.0f} | >120 s: {len(slow)} {sorted(slow)[-12:]}")
print("open trials with an empty agent dir (still starting):", len(waiting), sorted(waiting)[-15:])
