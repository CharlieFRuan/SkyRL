"""Overall numbers of eval rounds + per-task comparison.  eval_agg.py <label>=<aggregated_results.jsonl> ..."""
import json, sys, statistics as st
runs = {}
for a in sys.argv[1:]:
    label, path = a.split("=", 1)
    d = json.loads(open(path).readline())
    per = {}
    for k, v in d.items():
        if k.endswith("/avg_score") and "/all/" not in k:
            per[k.split("tbench21_4cpu_")[-1].split("/")[0]] = v
    overall = {k.split("/")[-1]: round(v, 4) for k, v in d.items() if k.startswith("eval/all/")}
    runs[label] = per
    print(f"{label}: tasks {len(per)} | mean of task avg {st.mean(per.values()):.4f} | solved at least once {sum(v > 0 for v in per.values())} ({sum(v > 0 for v in per.values()) / len(per):.4f}) | all: {overall}")
labels = list(runs)
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = runs[labels[i]], runs[labels[j]]
        common = sorted(set(a) & set(b))
        diff = [b[t] - a[t] for t in common]
        up = [(t, round(a[t], 2), round(b[t], 2)) for t in common if b[t] - a[t] > 0.3]
        down = [(t, round(a[t], 2), round(b[t], 2)) for t in common if a[t] - b[t] > 0.3]
        sd = st.pstdev(diff); se = sd / len(diff) ** 0.5
        print(f"{labels[j]} - {labels[i]}: {st.mean(diff):+.4f} (paired se {se:.4f}); tasks differing: {sum(d != 0 for d in diff)}; up by > 0.3: {len(up)}, down by > 0.3: {len(down)}")
        print("   up:", up[:12]); print("   down:", down[:12])
