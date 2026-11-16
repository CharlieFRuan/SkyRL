"""Capped replies in training vs eval, and how the loops begin (run ON the driver, run venv, PYTHONPATH=skycap/src).
    python loop_stats.py <run> [train_sample_every]"""
import collections, datetime, glob, json, os, re, sys
from pathlib import Path
from skycap import record

D = "/mnt/local_storage/charlie"; run = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 4
R = f"{D}/runs/{run}"
tb = set(os.listdir(f"{D}/data/tbench21_4cpu"))
FULL = 45.0  # a reply cut at the full 16,384 tokens takes over a minute; one cut by the context limit is short

stats = {"eval": collections.Counter(), "train": collections.Counter()}
by_hour = collections.defaultdict(collections.Counter); looped = {"eval": [], "train": []}
dirs = sorted(glob.glob(R + "/trials/*"))
for i, d in enumerate(dirs):
    name = os.path.basename(d); ph = "eval" if name.rsplit("__", 1)[0] in tb else "train"
    if ph == "train" and i % every: continue
    p = d + "/agent/timing.jsonl"
    if not os.path.exists(p) or not os.path.exists(d + "/result.json"): continue
    calls = full = short = 0; t0 = None
    for l in open(p):
        if '"llm"' not in l: continue
        try: e = json.loads(l)
        except Exception: continue
        if e.get("k") != "llm": continue
        t0 = t0 or e["t0"]; calls += 1
        if e.get("error") == "OutputLengthExceededError":
            if e["dt"] >= FULL: full += 1
            else: short += 1
    s = stats[ph]; s["trials"] += 1; s["calls"] += calls; s["full"] += full; s["short"] += short; s["trials_full"] += full > 0
    if ph == "train" and t0:
        h = datetime.datetime.fromtimestamp(t0).strftime("%m-%d %H"); b = by_hour[h]; b["trials"] += 1; b["calls"] += calls; b["full"] += full; b["tf"] += full > 0
    if full: looped[ph].append(d)
for ph in ("eval", "train"):
    s = stats[ph]
    print(f"{ph}: trials {s['trials']}, calls {s['calls']}, replies cut at the full cap {s['full']} ({100*s['full']/max(1,s['calls']):.2f}% of calls), "
          f"trials with one {s['trials_full']} ({100*s['trials_full']/max(1,s['trials']):.1f}%), replies cut short by the context limit {s['short']}")
print("training by start hour: trials, % of calls at the full cap, % of trials with one")
for h in sorted(by_hour):
    b = by_hour[h]; print("  %s  %5d  %.2f%%  %.1f%%" % (h, b["trials"], 100 * b["full"] / max(1, b["calls"]), 100 * b["tf"] / max(1, b["trials"])))

# How loops begin: the first repeated segment of each full-cap reply.
norm = lambda s: re.sub(r"\d+", "0", re.sub(r"\s+", " ", s)).strip()
def onset(text):
    body = text.split("<|im_start|>assistant", 1)[-1]
    think_end = body.find("</think>")
    segs = [m for m in re.finditer(r"[^\n.!?]{12,}[.!?\n]", body)]
    keys = [norm(m.group(0)) for m in segs]
    cnt = collections.Counter(keys)
    rep = [cnt[k] >= 4 for k in keys]
    for i in range(len(keys)):
        if rep[i] and sum(rep[i:i + 12]) >= 8:
            pos = segs[i].start(); first = segs[i].group(0).strip()
            return {"where": "thinking" if (think_end < 0 or pos < think_end) else "answer", "first": first, "frac": pos / max(1, len(body)),
                    "top": cnt.most_common(1)[0][0]}
    return None
def lead(s):
    s = re.sub(r"^[\s\\n#\-*>`\"']+", "", s.replace("\\n", " "))
    return " ".join(s.split()[:2]).rstrip(",:").lower() if s else ""
for ph in ("eval", "train"):
    first = collections.Counter(); where = collections.Counter(); n = 0; none = 0; fracs = []; ex = []
    for d in looped[ph][:400]:
        cfg = json.load(open(d + "/config.json")); m = re.search(r"/t/(tr_[0-9a-f]+)/", json.dumps(cfg))
        if not m or not os.path.exists(f"{R}/skycap/{m.group(1)}.json.zst"): continue
        try: tr = record.load(Path(R + "/skycap"), m.group(1))
        except Exception: continue
        for nd in tr.graph.nodes:
            if nd.author != "model" or nd.tokens is None or nd.tokens.text is None: continue
            if not any(c.finish_reason == "length" for c in nd.calls) or len(nd.tokens.token_ids) < 16000: continue
            text = nd.tokens.text if isinstance(nd.tokens.text, str) else bytes(nd.tokens.text).decode("utf-8", "replace")
            n += 1; o = onset(text)
            if o is None: none += 1; continue
            first[lead(o["first"])] += 1; where[o["where"]] += 1; fracs.append(o["frac"])
            if len(ex) < 6: ex.append(o["first"][:110].replace("\n", " "))
    fracs.sort()
    print(f"\n{ph}: full-cap replies read {n}; with a clear repeating block {n - none}; no exact repeat found {none}")
    if n - none:
        print("  loop is in:", dict(where), "| loop begins this far into the reply: median %.0f%%, p90 %.0f%%" % (100 * fracs[len(fracs) // 2], 100 * fracs[int(len(fracs) * .9)]))
        print("  first two words of the first repeated sentence:", [(k, v) for k, v in first.most_common(14)])
        print("  starts with wait:", sum(v for k, v in first.items() if k.startswith("wait")), "| actually:", sum(v for k, v in first.items() if k.startswith("actually")), "| let me / let's:", sum(v for k, v in first.items() if k.startswith("let")))
        for e in ex: print("   e.g.", e)
