#!/usr/bin/env python
"""CPU forward-pass sanity check: next-token logits of an HF export vs the base model on a short prompt.

Loads one model at a time (bf16, ~63 GB RAM each). Prints top-5 tokens, the logits' cosine similarity and the KL
between the two next-token distributions. A healthy RL checkpoint keeps cosine ~0.99+ and a small KL; a broken export
gives unrelated tokens / garbage.
"""

import argparse
import gc
import sys
import time

# Force transformers' pure-torch Mamba path: the CUDA-only causal_conv1d / mamba_ssm packages in the venv would
# otherwise be picked up and fail on CPU tensors.
for _pkg in ("causal_conv1d", "mamba_ssm"):
    sys.modules[_pkg] = None

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(path, ids):
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
    m.eval()
    with torch.no_grad():
        logits = m(input_ids=ids).logits[0].float()
    print(f"{path}: load+forward {time.time() - t:.0f}s", flush=True)
    del m
    gc.collect()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--base-hf", required=True)
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(args.base_hf)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    lb = run(args.base_hf, ids)
    le = run(args.export, ids)
    for name, l in (("base", lb), ("export", le)):
        top = torch.topk(l[-1], 5)
        print(name, [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
    cos = F.cosine_similarity(lb, le, dim=-1)
    kl = F.kl_div(F.log_softmax(le, -1), F.log_softmax(lb, -1), log_target=True, reduction="none").sum(-1)
    agree = (lb.argmax(-1) == le.argmax(-1)).float().mean()
    print(f"per-position logits cosine: min {cos.min():.4f} mean {cos.mean():.4f}")
    print(f"per-position KL(base||export): max {kl.max():.4f} mean {kl.mean():.4f}; argmax agreement {agree:.2f}")


if __name__ == "__main__":
    main()
