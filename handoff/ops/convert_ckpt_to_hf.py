#!/usr/bin/env python
"""Convert a SkyRL Megatron (torch_dist) checkpoint of Nemotron-3.5-Lightning (NemotronH) to an HF safetensors dir.

Single process, CPU only, no Megatron model build, no process group. It relies on how Megatron-Core writes
torch_dist checkpoints: every model parameter is saved under its *global* (TP/EP-unsharded) shape, so a plain
``torch.distributed.checkpoint`` load into full-size CPU tensors reshards from any TP/EP/PP layout. Optimizer state
(``chained_*``) is never read. The Megatron -> HF name/layout mapping mirrors megatron-bridge's ``NemotronHBridge``
(the same mapping weight sync uses to feed vLLM during training):

  embedding.word_embeddings.weight                 -> backbone.embeddings.weight
  output_layer.weight                              -> lm_head.weight
  decoder.final_norm.weight                        -> backbone.norm_f.weight
  Mamba layer i (decoder.layers.i.mixer.*):
    in_proj.layer_norm_weight                      -> backbone.layers.i.norm.weight
    in_proj.weight.{z,x,B,C,dt}  (concat dim 0)    -> backbone.layers.i.mixer.in_proj.weight
    conv1d.{weight,bias}.{x,B,C} (concat dim 0)    -> backbone.layers.i.mixer.conv1d.{weight,bias}
    A_log, D, dt_bias, norm.weight, out_proj.weight -> same name under backbone.layers.i.mixer
  MoE layer i (decoder.layers.i.*):
    pre_mlp_layernorm.weight                       -> backbone.layers.i.norm.weight
    mlp.router.weight                              -> backbone.layers.i.mixer.gate.weight
    mlp.router.expert_bias (fp32, kept fp32)       -> backbone.layers.i.mixer.gate.e_score_correction_bias
    mlp.experts.experts.linear_fc{1,2}.weight[e]   -> backbone.layers.i.mixer.experts.e.{up,down}_proj.weight
    mlp.shared_experts.linear_fc{1,2}.weight       -> backbone.layers.i.mixer.shared_experts.{up,down}_proj.weight
  Attention layer i (decoder.layers.i.self_attention.*):
    linear_qkv.layer_norm_weight                   -> backbone.layers.i.norm.weight
    linear_qkv.weight (per-group [q.., k, v] rows) -> backbone.layers.i.mixer.{q,k,v}_proj.weight
    linear_proj.weight                             -> backbone.layers.i.mixer.o_proj.weight

MTP (``mtp.*``) was disabled during training, so the checkpoint has no MTP weights; they are copied unchanged from
the base model (``--drop-mtp`` leaves them out). The output keeps the base model's shard layout, index, config,
tokenizer and chat template, and every tensor is checked against the base for identical name/shape/dtype.
"""

import argparse
import json
import os
import resource
import shutil
import struct
import sys
import time
from collections import defaultdict

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

WEIGHT_SUFFIXES = (".safetensors",)
# Small files copied from the base model dir (the checkpoint's policy/huggingface/config.json is re-serialized by
# transformers 5 -- e.g. layers_block_type "linear_attention" -- so the base model's originals are used).
SKIP_COPY = {"model.safetensors.index.json"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_st_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h


ST_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def build_plan(cfg, ckpt_keys):
    """Return {hf_name: (fn, [megatron keys])} for all non-MTP HF tensors; fn(list_of_tensors) -> hf tensor."""
    L = cfg["num_hidden_layers"]
    types = cfg["layers_block_type"]
    assert len(types) == L, (len(types), L)
    nh, ng, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    n_exp = cfg["n_routed_experts"]
    heads_per_group = nh // ng

    ident = lambda ts: ts[0]  # noqa: E731
    cat0 = lambda ts: torch.cat(ts, dim=0)  # noqa: E731

    def qkv_part(part):
        def fn(ts):
            w = ts[0]
            w = w.view(ng, (heads_per_group + 2) * hd, -1)
            q, k, v = torch.split(w, [heads_per_group * hd, hd, hd], dim=1)
            out = {"q": q, "k": k, "v": v}[part]
            return out.reshape(-1, w.shape[-1]).contiguous()

        return fn

    def expert(e):
        return lambda ts: ts[0][e]

    plan = {
        "backbone.embeddings.weight": (ident, ["embedding.word_embeddings.weight"]),
        "lm_head.weight": (ident, ["output_layer.weight"]),
        "backbone.norm_f.weight": (ident, ["decoder.final_norm.weight"]),
    }
    for i, t in enumerate(types):
        m, h = f"decoder.layers.{i}.", f"backbone.layers.{i}."
        if t == "mamba":
            plan[h + "norm.weight"] = (ident, [m + "mixer.in_proj.layer_norm_weight"])
            plan[h + "mixer.in_proj.weight"] = (cat0, [m + f"mixer.in_proj.weight.{p}" for p in "z x B C dt".split()])
            for p in ("weight", "bias"):
                plan[h + f"mixer.conv1d.{p}"] = (cat0, [m + f"mixer.conv1d.{p}.{s}" for s in "xBC"])
            for p in ("A_log", "D", "dt_bias", "norm.weight", "out_proj.weight"):
                plan[h + f"mixer.{p}"] = (ident, [m + f"mixer.{p}"])
        elif t == "moe":
            plan[h + "norm.weight"] = (ident, [m + "pre_mlp_layernorm.weight"])
            plan[h + "mixer.gate.weight"] = (ident, [m + "mlp.router.weight"])
            plan[h + "mixer.gate.e_score_correction_bias"] = (ident, [m + "mlp.router.expert_bias"])
            for fc, proj in (("linear_fc1", "up_proj"), ("linear_fc2", "down_proj")):
                plan[h + f"mixer.shared_experts.{proj}.weight"] = (ident, [m + f"mlp.shared_experts.{fc}.weight"])
                for e in range(n_exp):
                    plan[h + f"mixer.experts.{e}.{proj}.weight"] = (expert(e), [m + f"mlp.experts.experts.{fc}.weight"])
        elif t == "attention":
            plan[h + "norm.weight"] = (ident, [m + "self_attention.linear_qkv.layer_norm_weight"])
            for part in "qkv":
                plan[h + f"mixer.{part}_proj.weight"] = (qkv_part(part), [m + "self_attention.linear_qkv.weight"])
            plan[h + "mixer.o_proj.weight"] = (ident, [m + "self_attention.linear_proj.weight"])
        else:
            raise ValueError(f"unsupported layer type {t!r} at layer {i}")

    used = {k for _, ks in plan.values() for k in ks}
    missing = sorted(used - ckpt_keys)
    unused = sorted(ckpt_keys - used)
    if missing or unused:
        raise SystemExit(
            f"checkpoint/mapping mismatch: {len(missing)} mapped keys missing from ckpt (e.g. {missing[:5]}), "
            f"{len(unused)} ckpt model keys unmapped (e.g. {unused[:5]})"
        )
    return plan


def load_megatron(reader_path, md, keys):
    """Load full (global) tensors for `keys` from the torch_dist checkpoint on CPU, single process."""
    sd = {}
    for k in keys:
        m = md.state_dict_metadata[k]
        sd[k] = torch.empty(m.size, dtype=m.properties.dtype)
    dcp.load(sd, storage_reader=FileSystemReader(reader_path), no_dist=True)
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="global_step_N dir (or its policy/ subdir)")
    ap.add_argument("--base-hf", required=True, help="base HF model dir (config, tokenizer, safetensors index)")
    ap.add_argument("--out", required=True, help="output HF dir (must not exist or be empty)")
    ap.add_argument("--drop-mtp", action="store_true", help="omit mtp.* tensors instead of copying them from base")
    args = ap.parse_args()

    t0 = time.time()
    policy = args.ckpt
    if os.path.isdir(os.path.join(policy, "policy")):
        policy = os.path.join(policy, "policy")
    with open(os.path.join(policy, "metadata.json")) as f:
        backend = json.load(f)
    assert backend.get("sharded_backend") == "torch_dist", backend
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"output dir {args.out} exists and is not empty")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.base_hf, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("model_type") == "nemotron_h", cfg.get("model_type")
    with open(os.path.join(args.base_hf, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    md = FileSystemReader(policy).read_metadata()
    ckpt_keys = {
        k
        for k, v in md.state_dict_metadata.items()
        if isinstance(v, TensorStorageMetadata)
        and not k.startswith(("chained_", "optimizer"))
        and "_extra_state" not in k
    }
    log(f"checkpoint {policy}: {len(md.state_dict_metadata)} entries, {len(ckpt_keys)} model tensors")
    if any(k.startswith("mtp.") for k in ckpt_keys):
        raise SystemExit("checkpoint contains MTP weights; this converter only handles MTP-disabled training runs")

    plan = build_plan(cfg, ckpt_keys)
    hf_from_base = sorted(k for k in weight_map if k not in plan)
    bad = [k for k in hf_from_base if not k.startswith("mtp.")]
    if bad:
        raise SystemExit(f"base HF tensors without a mapping: {bad[:10]}")
    extra = sorted(set(plan) - set(weight_map))
    if extra:
        raise SystemExit(f"mapped HF names not in base index: {extra[:10]}")
    log(f"{len(plan)} HF tensors from the checkpoint, {len(hf_from_base)} mtp.* tensors "
        f"{'dropped' if args.drop_mtp else 'copied from base'}")

    by_file = defaultdict(list)
    for k, f in weight_map.items():
        by_file[f].append(k)

    new_map = {}
    total_bytes = 0
    for fi, fname in enumerate(sorted(by_file)):
        names = by_file[fname]
        base_path = os.path.join(args.base_hf, fname)
        hdr = read_st_header(base_path)
        need = sorted({mk for n in names if n in plan for mk in plan[n][1]})
        t1 = time.time()
        msd = load_megatron(policy, md, need) if need else {}
        out = {}
        mtp_names = [n for n in names if n not in plan]
        if mtp_names and not args.drop_mtp:
            with safe_open(base_path, framework="pt") as sf:
                for n in mtp_names:
                    out[n] = sf.get_tensor(n)
        for n in names:
            if n in plan:
                fn, mks = plan[n]
                out[n] = fn([msd[k] for k in mks]).contiguous()
        for n, t in out.items():
            exp_shape, exp_dtype = tuple(hdr[n]["shape"]), ST_DTYPES[hdr[n]["dtype"]]
            if tuple(t.shape) != exp_shape or t.dtype != exp_dtype:
                raise SystemExit(f"{n}: got {tuple(t.shape)} {t.dtype}, base has {exp_shape} {exp_dtype}")
        # Shared storage (expert slices of one grouped tensor) must be cloned for safetensors.
        out = {n: t.clone() if n in plan and t.untyped_storage().nbytes() != t.nbytes else t for n, t in out.items()}
        tmp = os.path.join(args.out, fname + ".tmp")
        save_file(out, tmp, metadata={"format": "pt"})
        os.chmod(tmp, 0o644)
        os.replace(tmp, os.path.join(args.out, fname))
        for n, t in out.items():
            new_map[n] = fname
            total_bytes += t.nbytes
        log(f"[{fi + 1}/{len(by_file)}] {fname}: {len(out)} tensors ({len(need)} ckpt tensors) in {time.time() - t1:.0f}s, "
            f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
        del msd, out

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(new_map.items()))}, f, indent=2)
    for fn in sorted(os.listdir(args.base_hf)):
        p = os.path.join(args.base_hf, fn)
        if os.path.isfile(p) and not fn.startswith(".") and not fn.endswith(WEIGHT_SUFFIXES) and fn not in SKIP_COPY:
            shutil.copy2(p, os.path.join(args.out, fn))
    with open(os.path.join(args.out, "skyrl_conversion.json"), "w") as f:
        json.dump(
            {
                "source_checkpoint": os.path.abspath(policy),
                "base_hf": os.path.abspath(args.base_hf),
                "mtp": "dropped" if args.drop_mtp else "copied from base (not trained)",
                "converter": os.path.abspath(__file__),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    log(f"done: {args.out} ({total_bytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s, peak RSS {peak:.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
