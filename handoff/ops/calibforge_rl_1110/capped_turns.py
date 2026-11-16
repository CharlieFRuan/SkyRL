"""Capped (finish_reason=length) turns in a run's skycap records: did the next call extend the truncated reply's
sampled tokens (bridged) or fork?   python capped_turns.py <run>   (driver node, the run's venv, PYTHONPATH=skycap/src)"""
import collections, sys
from pathlib import Path
from skycap import record

rd = Path(f"/mnt/local_storage/charlie/runs/{sys.argv[1]}/skycap")
ids = list(record.list_ids(rd))
tot = collections.Counter(); shown = 0
for tid in ids:
    try: tr = record.load(rd, tid)
    except Exception as e:  # noqa: BLE001
        tot["unreadable"] += 1; continue
    g = tr.graph; nodes = list(g.nodes)
    kids = collections.defaultdict(list)
    for n in nodes:
        if n.parent is not None: kids[n.parent].append(n)
    for n in nodes:
        cut = [c for c in n.calls if c.finish_reason == "length"]
        if not cut: continue
        tot["capped turns"] += 1
        # the model nodes that follow this one
        nxt, frontier = [], list(kids[n.id])
        while frontier:
            m = frontier.pop()
            if m.author == "model" and m.calls: nxt.append(m)
            else: frontier += kids[m.id]
        if not nxt: tot["capped turn was the trajectory's last call"] += 1
        for m in nxt:
            for c in m.calls:
                tot[f"next call bridged={c.bridged}"] += 1
        if shown < 6:
            shown += 1
            leaves = sum(1 for x in nodes if not kids[x.id])
            print(tid, (tr.meta or {}).get("task", "").rsplit("/", 1)[-1], "| capped:", cut[0].usage, "| next:",
                  [(c.bridged, c.finish_reason, c.usage) for m in nxt for c in m.calls][:2],
                  "| unbridged calls in trajectory:", g.unbridged_calls(), "| leaves:", leaves, "| model calls:", sum(len(x.calls) for x in nodes))
print("records", len(ids), dict(tot))
