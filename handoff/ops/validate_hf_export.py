#!/usr/bin/env python
"""Validate an HF export against the base model: same tensor names/shapes/dtypes, small relative deltas.

For every tensor: rel = ||new - base|| / ||base||. A wrong layout (QKV split, Mamba in_proj order, expert index) shows
up as rel ~ 1 (or more), training deltas as rel << 0.1. Prints per-pattern stats and exits non-zero on failure.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--base-hf", required=True)
    ap.add_argument("--max-rel", type=float, default=0.1, help="fail if any tensor's relative diff exceeds this")
    args = ap.parse_args()

    idx_new = json.load(open(os.path.join(args.export, "model.safetensors.index.json")))["weight_map"]
    idx_base = json.load(open(os.path.join(args.base_hf, "model.safetensors.index.json")))["weight_map"]
    ok = True
    if set(idx_new) != set(idx_base):
        print(f"KEY MISMATCH: only in export {sorted(set(idx_new) - set(idx_base))[:10]}, "
              f"only in base {sorted(set(idx_base) - set(idx_new))[:10]}")
        ok = False

    stats = defaultdict(list)  # pattern -> list of (rel, name, changed_frac)
    worst = []
    by_file = defaultdict(list)
    for k in sorted(set(idx_new) & set(idx_base)):
        by_file[(idx_new[k], idx_base[k])].append(k)
    for (fn_new, fn_base), names in sorted(by_file.items()):
        with safe_open(os.path.join(args.export, fn_new), "pt") as fa, safe_open(os.path.join(args.base_hf, fn_base), "pt") as fb:
            for n in names:
                a, b = fa.get_tensor(n), fb.get_tensor(n)
                if a.shape != b.shape or a.dtype != b.dtype:
                    print(f"SHAPE/DTYPE MISMATCH {n}: {a.shape} {a.dtype} vs base {b.shape} {b.dtype}")
                    ok = False
                    continue
                af, bf = a.float(), b.float()
                rel = ((af - bf).norm() / bf.norm().clamp_min(1e-12)).item()
                changed = (a != b).float().mean().item()
                pat = re.sub(r"\.\d+\.", ".N.", n)
                stats[pat].append((rel, n, changed))
                worst.append((rel, n))
    print(f"{'pattern':62s} {'n':>5s} {'rel_mean':>9s} {'rel_max':>9s} {'chg_frac':>8s} {'n_identical':>11s}")
    for pat, v in sorted(stats.items()):
        rels = [r for r, _, _ in v]
        chg = sum(c for _, _, c in v) / len(v)
        ident = sum(1 for _, _, c in v if c == 0)
        print(f"{pat:62s} {len(v):5d} {sum(rels) / len(rels):9.2e} {max(rels):9.2e} {chg:8.3f} {ident:11d}")
    worst.sort(reverse=True)
    print("worst 5:", [(f"{r:.3e}", n) for r, n in worst[:5]])
    bad = [(r, n) for r, n in worst if r > args.max_rel]
    if bad:
        print(f"FAIL: {len(bad)} tensors with rel diff > {args.max_rel}: {bad[:10]}")
        ok = False
    nonmtp_ident = [n for r, n in worst if r == 0 and not n.startswith("mtp.")]
    print(f"non-mtp tensors bit-identical to base: {len(nonmtp_ident)} (e.g. {nonmtp_ident[:5]})")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
