"""Re-score SkyCap-recorded trajectories with a plain vLLM endpoint's prefill (prompt_logprobs), and compare with the
logprobs recorded when the tokens were sampled. Separates SkyCap bookkeeping and vLLM decode-vs-prefill numerics from
the vLLM-vs-Megatron gap.

    python rescore_skycap.py <skycap record dir> <vllm base url> [max trajectories] [step]

Only trajectories whose meta step equals [step] when given (e.g. 1: sampled by the base weights).

Prints mean |recorded - rescored| over sampled tokens, overall and by position bucket.
"""

import collections
import json
import sys
import urllib.request
from pathlib import Path

from skycap import record

rd, url = Path(sys.argv[1]), sys.argv[2].rstrip("/")
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
step = int(sys.argv[4]) if len(sys.argv) > 4 else None
model = json.load(urllib.request.urlopen(f"{url}/v1/models"))["data"][0]["id"]
buckets = collections.defaultdict(lambda: [0.0, 0])
total = [0.0, 0]
worst = []
done = 0
for tid in record.list_ids(rd):
    if done >= limit:
        break
    trajectory = record.load(rd, tid)
    if step is not None and trajectory.meta.get("step") != step:
        continue
    done += 1
    g = trajectory.graph
    nodes = {n.id: n for n in g}
    path = max(g.paths(), key=len)  # the longest root-to-leaf path
    ids, recorded = [], []  # the full sequence, and per position the recorded logprob of a sampled token (or None)
    for nid in path:
        t = nodes[nid].tokens
        if t is None:
            continue
        start = t.sampled_start if t.sampled_start is not None else len(t.token_ids)
        for k, tok in enumerate(t.token_ids):
            ids.append(tok)
            lp = t.logprobs[k] if (t.logprobs is not None and k >= start) else None
            recorded.append(lp)
    body = {"model": model, "prompt": ids, "max_tokens": 1, "prompt_logprobs": 0, "temperature": 1.0}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=1800))
    plp = out["choices"][0]["prompt_logprobs"]
    diffs = []
    for pos, (tok, lp) in enumerate(zip(ids, recorded)):
        if lp is None or pos == 0 or plp[pos] is None:
            continue
        got = plp[pos].get(str(tok)) or plp[pos].get(tok)
        if got is None:
            continue
        d = abs(lp - got["logprob"])
        diffs.append(d)
        b = min(pos // 8192, 15)
        buckets[b][0] += d
        buckets[b][1] += 1
        total[0] += d
        total[1] += 1
        worst.append((d, tid, pos, lp, got["logprob"]))
    print(f"{tid}: {len(ids)} tokens, {len(diffs)} sampled, mean |d| {sum(diffs) / max(1, len(diffs)):.4f}", flush=True)
print(f"ALL: mean |recorded - prefill| = {total[0] / max(1, total[1]):.4f} over {total[1]} sampled tokens")
for b in sorted(buckets):
    s, n = buckets[b]
    print(f"  positions {b * 8}k-{(b + 1) * 8}k: {s / n:.4f} (n={n})")
for d, tid, pos, a, b in sorted(worst, reverse=True)[:5]:
    print(f"  worst {d:.2f} {tid} pos {pos}: recorded {a:.3f} prefill {b:.3f}")
