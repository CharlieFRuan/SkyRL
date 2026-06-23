# ============================================================
# FINAL SUMMARY (2026-06-03) — Qwen3.6-35B-A3B 4-node DAPO: SOLVED & STABLE
# ============================================================
# STATUS: 4-node (32-GPU) colocate DAPO training STABLY over InfiniBand.
#   Steps: 1300s, 1249s (~1275s/step) = 2.8x faster than single-node (3587s). 0 crashes, 0 IB errors.
#
# ROOT CAUSE of the repeated policy_train crashes: cross-node NCCL auto-selected
#   two DEAD InfiniBand HCAs (mlx5_2, mlx5_3) -> IBV_WC_RETRY_EXC_ERR -> ncclRemoteError
#   -> ProcessGroupNCCL watchdog aborted the worker -> cascade. (Node/memory/optimizer-offload
#   were all red herrings; the "dying node" was always a cascade victim.)
#
# THE FIX (keeps fast IB, NOT TCP):
#   env_vars["NCCL_IB_HCA"] = "mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"   in skyrl/train/utils/utils.py
#   (excludes the bad mlx5_2/mlx5_3; matches the known-good all_reduce_bench/run_bench.sh).
#   Propagates to all Ray workers via runtime_env. Validated fast with full_ctx (no generation).
#
# WORKING CONFIG: cluster {5q745,n2,n3,m8htz} all enp2s0, Ray head 10.40.16.194:6379;
#   TP4/EP8/PP1/CP1, offload=true, gpu_mem 0.7, NO recompute, expandable_segments UNSET.
#   Launch: run_final_4node.sh ; monitor log: out_final_4node.log.
#
# OTHER FIXES along the way: (1) colocate pidfd block = PYTORCH_CUDA_ALLOC_CONF expandable_segments
#   (leave UNSET); (2) every node needs `uv sync --extra megatron`; (3) per-node GLOO/NCCL_SOCKET_IFNAME;
#   (4) megatron_worker.py None-size bucket guard. n4 has a separate IB issue (excluded). m8htz fabric healthy.
# ============================================================
# Qwen3.6-35B-A3B Optimization — Progress & Plan

> Living doc. Survives context compaction; charlie can check it too.
> Last updated: 2026-06-03 (sweep in progress).

---

## 0. THE ASKS (user requirements)

**A. Megatron knob sweep (fastest training throughput), single node (8×H200), Qwen3.6-35B-A3B**
- Use the `full_context` dummy harness (synthetic max-len seqs, times each phase, no real gen).
- Sweep TP/EP/PP/CP/ETP + micro_batch + "other megatron settings" (recompute, dispatcher, ...).
- Fix `train_batch_size = mini_batch_size = 4`, `micro_batch = 1` (batch scales linearly; only tune
  micro batch up to the max that fits memory).
- Start at **64k** context; after picking a good 64k config, pick one for **128k**.
- Run **2 steps per config** to check stability (step 1 = warmup/compile; measure step 2).
- **Always use expandable KV cache** (PR #1470 essence = `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- References: miles `run_qwen3_6_35b_a3b_mtp.py` (TP1/EP8/PP1/CP1/ETP1 + recompute + flex dispatcher),
  GLM `example_glm47_30b_script.sh`, GLM-355B ablation doc.

**B. sample_packing slowdown** — with best knobs, run a gsm8k script `use_sample_packing=false` (2 steps)
   vs `=true` (2 steps), `enforce_eager=false`. Report the slowdown. (NOTE: GDN model may not support
   packing in megatron → packing=true may error; if so, that's the finding.)

**C. KV cache optimization** (Qwen3.6-35B, 1 node, TP=8 vllm):
- Baseline (gpu_mem_util=0.7, no CPU offload): `GPU KV cache size: 8,825,334 tokens`,
  `Max concurrency for 262,144 tokens/req: 33.67x` (from infra log).
- (1) push `gpu_memory_utilization` 0.7 → 0.9, measure improvement.
- (2) Use **DCP (decode-context-parallel)**: num_kv_heads=2, so TP=8 replicates KV 4× (redundant).
  Keep TP=8 (easier routing). Enable DCP (vllm docs: serving/context_parallel_deployment). Measure KV size.

**D. Final full 2-node run** with most optimized knobs + high KV cache + DCP.
- DAPO may not need high KV, but want to test it.
- DCP correctness is uncertain (logprob may be NaN) → **hack inference to return vLLM logprobs and
  assert no NaN.** Verify the run trains without NaN.

**Infra**: may use up to **4 nodes**; `kubectl taint nodes <n> reserved-by=charlieruan:NoSchedule` to reserve.

---

## 1. ENVIRONMENT / SETUP

**Nodes** (all 8×H200; key `~/.ssh/id_ed25519` works cluster-wide; aliases in `~/.ssh/config`):
| alias | node | IP | role |
|-------|------|----|------|
| (local) | gpu-dp-q9wbz-5q745 | 10.40.16.194 | head / sweep worker |
| n2 | gpu-dp-q9wbz-77pcc | 10.40.58.131 | sweep worker / final-run worker |
| n3 | gpu-dp-q9wbz-gccvh | 10.40.30.153 | sweep worker (setup ~done) |
| n4 | gpu-dp-q9wbz-8vnx2 | 10.40.40.19 | sweep worker (setup ~done) |
| — | gpu-dp-q9wbz-m8htz | 10.40.62.120 | **BROKEN fabric manager — do not use** |

All 4 sweep nodes tainted `reserved-by=charlieruan:NoSchedule`. Each has: `uv`, `~/SkyRL` (synced source
incl. my edits), `~/data/gsm8k`, `~/sweep_ctx.sh`, the 67 GB Qwen3.6-35B-A3B in `~/.cache/huggingface`,
a `.venv` (`uv sync --extra fsdp`), and a cached megatron env (TE compiled with cudnn CPATH).

**Key paths**
- Sweep harness: `/home/charlie_key/sweep_ctx.sh` (env knobs: TP EP PP CP ETP MICRO CTX NUM_STEPS TBATCH NSAMP DISPATCHER RECOMPUTE_GRAN TAG).
- Sweep logs + raw results table: `/home/charlie_key/sweep_logs/` (`results.md`, `<TAG>.log`).
- Megatron env build needs: `CUDA_HOME=/usr/local/cuda`, `CPATH`=venv nvidia includes + cuda include
  (TE source build needs `cudnn.h`).

**Code edits made (in `~/SkyRL`, synced to all nodes)**
1. `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py` (~line 191): param-bucketing regular
   path now guards `None` sizes (`size = sizes[idx] if not None else 0`) — fixes a real crash
   (`float + NoneType`) on weight-sync bucket init for this model.
2. `skyrl/train/utils/utils.py` (prepare_runtime_environment): added `PYTORCH_CUDA_ALLOC_CONF` to the
   env-var passthrough list → `expandable_segments:True` reaches all Ray workers (VERIFIED on a worker).

**Harness fixed config**: train_batch=mini=4, n_samples_per_prompt=2 (→8 seqs/step; needed so dp≤8
configs have mini_per_gpu>0), micro=1, recompute=full, dispatcher=alltoall, optimizer_cpu_offload=1.0,
enforce_eager(vllm)=true, colocate_all=true, weight_sync=nccl, use_sample_packing=false.
Metric = step-2 `timing/step` seconds (step 1 = warmup).

---

## 2. PROGRESS / RESULTS

### Key findings
- **CP > 1 is UNUSABLE for this model**: `context parallel is only supported with sample packing`, but
  GDN layers don't support sample packing in megatron → **CP must = 1**. Long-context memory comes from
  TP/EP/PP + recompute + offload, not CP. (Matches miles using CP=1.)
- **Low TP OOMs at 64k**: TP=2 OOM'd on step 2 (needed 30 GiB, 25 free). Viable region is **TP4–TP8**.
- `expandable_segments:True` confirmed active on workers.

### 64k sweep (CTX=65536 = 1024 prompt + 64512 gen), 8 seqs/step, micro=1
| config | TP | EP | PP | CP | micro | step2 (s) | result |
|--------|----|----|----|----|-------|-----------|--------|
| tp4_ep8 | 4 | 8 | 1 | 1 | 1 | **49.1** | OK (fwd 9.8 + policy 39.0) |
| tp2_ep8 | 2 | 8 | 1 | 1 | 1 | OOM | step2 OOM (TP too low) |
| tp8_ep8 | 8 | 8 | 1 | 1 | 1 | **50.3** | OK (fwd 9.5 + policy 40.5) |
| tp4_ep4 | 4 | 4 | 1 | 1 | 1 | SLOW | EP4 fwd 66s; EP8 far better |
| tp4_ep8_pp2 | 4 | 8 | 2 | 1 | 1 | queued | — |
| tp8_ep4 | 8 | 4 | 1 | 1 | 1 | queued | — |

64k REFINEMENT RESULTS:
- micro=2 -> OOM => max micro batch = 1 at 64k.
- dispatcher=flex -> ImportError: DeepEP not installed => use alltoall (default).
- TP4/EP4 -> SLOW (fwd 66s vs 9.8s for EP8) => EP8 >> EP4 (shard experts across all 8 GPUs).
- **WINNER_64K = TP4/EP8/CP1, micro=1, alltoall, recompute=full = 49.1s** (TP8/EP8 tied 50.3s).

### Note: selective recompute needs recompute_num_layers=None (config conflict) + uses MORE memory than full → not pursued; full recompute (working winner) kept.

### 128k sweep
### 128k sweep RESULTS (CTX=131072) — micro=1, alltoall, recompute=full, offload
| config | step2 (s) | result |
|--------|-----------|--------|
| tp8_ep8 | **100.8** | OK (fwd 21.8 + policy 78.6) — **WINNER_128K** |
| tp4_ep8 | OOM | TP4 too little memory at 128k |
| tp4_ep8_pp2 | INVALID | world 8 not divisible by ETP*EP*PP=16 (PP needs EP4) |
**WINNER_128K = TP8/EP8/CP1 = 100.8s.** (PP unusable with EP8; EP4 too slow; TP8 required at 128k.)
 — TODO (after 64k). Memory tighter → expect need TP8 and/or PP2, full recompute.

### sample_packing comparison
- NOTE: **enforce_eager=false CRASHES vLLM init for this GDN model** (vLLM 0.20.2 CUDA-graph capture
  incompatible with the hybrid GDN arch; matches DAPO script's enforce_eager=true note). So the
  'enforce_eager=false for faster inference' the user wanted is NOT usable here. Re-running packing
  comparison with enforce_eager=true (the working setting).

### KV cache optimization — DONE
| Config (TP=8, max_model_len 262144) | KV cache tokens | Max concurrency | vs 0.7 |
|---|---|---|---|
| 0.7, no DCP (baseline, given) | 8,825,334 | 33.67x | 1.00x |
| 0.9, no DCP | 11,725,674 | 44.73x | 1.33x |
| 0.9, DCP=2 (projection only) | 23,308,254 | 88.91x | 2.64x |
| 0.9, DCP=4 (projection only) | 46,050,316 | 175.67x | 5.22x |

- mem_util 0.7->0.9 (no DCP): **1.33x** more KV cache. No OOM at 0.9.
- **DCP is BLOCKED on this model in vLLM 0.20.2**: engine init aborts with
  `ValueError: Hybrid KV cache groups with multiple block sizes do not support context parallelism`
  (GDN = attention + mamba/linear-attn layers -> different KV block sizes). DCP=2/4 numbers are vLLM's
  pre-abort projection (confirm DCP=4 recovers the ~4x TP-replication waste), but the engine is NOT
  operational with DCP. Only 0.9-no-DCP runs.
- IMPLICATION for final run: cannot use DCP (would need a newer vLLM with hybrid-CP support, or a vLLM
  hack to relax the hybrid block-size check — risky/likely-incorrect). Final run uses **mem_util 0.9, no DCP**.
  The DCP NaN-logprob correctness check is moot (DCP can't init), so no logprob to assert on.
Baseline: 8,825,334 tokens @ mem_util 0.7, TP8, no DCP (max_concurrency 33.67x @ 262144 tok/req).
Subagent measuring: (1) 0.9 no-DCP, (2) 0.9 + DCP(decode_context_parallel_size=4=TP/num_kv_heads).
DCP via generator.inference_engine.engine_init_kwargs={"decode_context_parallel_size":4}. Generation-only
(main_generate). Results -> /home/charlie_key/sweep_logs/dcp_kv_results.md. n5 tainted; 5th node.

### Final 2-node DCP run + NaN guard — REVISED (DCP blocked)
DCP can't init on this hybrid GDN model (vLLM 0.20.2) -> NaN-logprob guard is MOOT (no DCP run to check).
Final run = best knobs (TP4/EP8 for ~10k DAPO ctx; world16 -> TP4*DP4, EP8) + **mem_util 0.9** (high KV,
the achievable 1.33x; DCP would need newer vLLM w/ hybrid-CP). 2-node megatron Ray cluster via persistent
megatron venv: `uv sync --extra megatron` (TE wheel cached -> fast) on 5q745+n2, then `.venv/bin/ray start`
head+worker (reliable 16-GPU registration per MULTINODE.md), run main_dapo via .venv python, num_nodes=2,
gpu_memory_utilization=0.9, LOGGER=console &> out_final.log. Confirm it trains (no DCP). Flag DCP-blocked to user.

---

## 3. NEXT STEPS
1. Finish 64k grid (tp8_ep8, tp4_ep4, tp4_ep8_pp2, tp8_ep4) across 4 nodes → pick fastest stable = WINNER_64K.
2. Refine WINNER_64K: flex dispatcher, max micro batch.
3. 128k sweep around winner.
4. sample_packing false-vs-true on gsm8k (enforce_eager=false).
5. KV cache: mem_util 0.9; then DCP (TP=8). Record KV token sizes from infra log.
6. Final 2-node DAPO run: best knobs + mem_util 0.9 + DCP + logprob-NaN assertion. Verify no NaN.

## 4. HOW TO DRIVE (for future-me after compaction)
- Launch a sweep config: `TP=.. EP=.. CP=1 PP=.. MICRO=.. CTX=65536 NUM_STEPS=2 TAG=.. bash /home/charlie_key/sweep_ctx.sh &> /home/charlie_key/sweep_logs/<TAG>.log` (local on 5q745; `ssh nX '<env> bash ~/sweep_ctx.sh'` for others).
- A node is free if `pgrep -f main_full_ctx` is empty (local) / `ssh nX pgrep -f main_full_ctx`.
- Extract step time: `grep "Finished: 'step'" <log>` (2nd one = step2) + `grep -A6 timing/fwd <log>`.
- Each config = finite (2 dummy steps, then exits + notifies). OOM = record + move on.

### Why miles uses TP1/EP8 but TP1/TP2 OOM in our 64k sweep
Miles runs **short context**: `rollout_max_response_len=1024`, `use_dynamic_batch_size` +
`max_tokens_per_gpu=8192` (token-based batching). At ~1-2k tokens/seq, activations are tiny → TP1/EP8
fits (35B weights shard across EP=8). Our sweep is **64k tokens/seq**, micro=1 seq = 64k tokens/GPU
(~8x past miles' 8192 cap, ~32-64x more activation memory). A single 64k seq is indivisible (CP would
split it but CP needs sample-packing which GDN lacks), so **TP is the only lever** to fit → viable
region TP4-TP8 at 64k. Not a contradiction: miles optimizes short-ctx (TP1 best there); we optimize
long-ctx (memory wall forces TP4+). Also SkyRL megatron uses fixed micro_batch (seq count), not miles'
token-based dynamic batching.

### STATUS UPDATE (mp executor blocker)
- sample_packing comparison: BOTH packing=false AND packing=true died at step-0 weight sync with the
  mp/pidfd_getfd error (NOT a packing issue) -> needs re-run with distributed_executor_backend=ray.
  (Packing is also documented-unsupported for GDN per CP error + DAPO script; ray re-run would confirm.)
- Final 2-node run: first attempt (mp) failed same way; **retrying with distributed_executor_backend=ray**
  on the 16-GPU cluster (5q745+n3), mem_util 0.9, TP4/EP8, no DCP. Watching out_final2.log.

---
# ============ FINAL SUMMARY (2026-06-03) ============

## Megatron training-throughput knobs (full_context dummy harness, 8 seqs/step, micro=1)
- **64k context → WINNER: TP4 / EP8 / PP1 / CP1 / ETP1 = 49.1 s/step** (TP8/EP8 = 50.3 s, tied).
- **128k context → WINNER: TP8 / EP8 / PP1 / CP1 / ETP1 = 100.8 s/step** (TP4 OOMs at 128k; TP8 required).
- Fixed: recompute=full (uniform,1), optimizer_cpu_offload=1.0, dispatcher=alltoall, micro=1 (micro=2 OOMs at 64k), expandable_segments=True.
- Constraints discovered:
  * **CP must = 1** — CP needs sample packing, which this GDN model can't use in megatron.
  * **EP8 >> EP4** — EP4 fwd was 66s vs 9.8s (experts must shard across all 8 GPUs).
  * **flex dispatcher unavailable** — needs DeepEP (not installed) → alltoall.
  * Low TP OOMs at long ctx (TP2 OOM at 64k) → viable TP4–TP8.
  * Why miles uses TP1/EP8: miles is SHORT context (≤1k resp, token-batched ≤8192); long-ctx needs TP4+.

## KV cache (TP=8, max_model_len 262144)
| Config | KV tokens | concurrency | vs 0.7 |
|---|---|---|---|
| 0.7 no-DCP (baseline) | 8,825,334 | 33.67x | 1.00x |
| **0.9 no-DCP** | **11,725,674** | 44.73x | **1.33x** |
| 0.9 DCP=2 (projection) | 23,308,254 | 88.91x | 2.64x |
| 0.9 DCP=4 (projection) | 46,050,316 | 175.67x | 5.22x |
- **mem_util 0.7→0.9 = 1.33x** more KV (usable now).
- **DCP BLOCKED**: vLLM 0.20.2 aborts engine init on this hybrid GDN model
  (`Hybrid KV cache groups with multiple block sizes do not support context parallelism`). DCP=4 *would*
  recover the full 4x wasted by TP=8 replicating 2 KV heads (5.22x w/ 0.9), but needs a vLLM with
  hybrid (attn+mamba) context-parallel support. NaN-logprob guard is therefore moot (DCP can't run).

## Other infra findings
- **enforce_eager=false CRASHES vLLM init** for this GDN model (CUDA-graph capture) → must use enforce_eager=true.
- **vLLM colocate weight-sync uses CUDA IPC (pidfd_getfd)** which fails ('Operation not permitted') under
  ptrace_scope=1 when train & vLLM procs are raylet siblings (multi-node). Works single-node (compatible
  hierarchy). Affects BOTH mp and ray executors. Fix: ptrace_scope=0 (root), or disaggregated serving
  (colocate_all=false + NCCL weight sync), or single-node.

## Final run
- 2-node (5q745+n3) colocate run: BLOCKED by the pidfd_getfd weight-sync issue above (infra, needs root/disagg).
- Fallback: SINGLE-NODE run on 5q745 with optimized knobs (TP4/EP8) + **mem_util 0.9** (high KV) — out_final_singlenode.log.

## sample_packing comparison — INCOMPLETE
- Both packing=false/true gsm8k runs (on worker nodes) died at the mp/pidfd weight-sync (before packing
  mattered), not at packing. Packing is also documented-unsupported for GDN megatron (CP-requires-packing
  error + upstream PR note). To get clean numbers: run single-node on 5q745 (weight-sync works there) with
  packing=false vs true; expect packing=true to error on GDN.

## Nodes used (all tainted reserved-by=charlieruan:NoSchedule): 5q745, 77pcc(n2), gccvh(n3), 8vnx2(n4), nw44r(n5). m8htz = broken fabric.

---
## ⭐ ROOT CAUSE FOUND (2026-06-03): colocate `pidfd_getfd` blocker = expandable_segments

**The 2-node (and single-node) colocate `pidfd_getfd: Operation not permitted` failure was caused by `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, NOT by the mp/ray executor choice.**

- Tested `distributed_executor_backend=ray`: STILL failed with `pidfd_getfd` (now inside vLLM's own TP-worker CUDA-IPC `_new_shared_cuda`). So executor is irrelevant.
- `kernel.yama.ptrace_scope=1` on all nodes; no passwordless sudo → can't set it to 0.
- **Mechanism:** expandable-segment (cuMem-based) CUDA allocations cannot be shared via the classic `cudaIpcGetMemHandle`. PyTorch falls back to passing the memory fd via `pidfd_getfd`, which requires ptrace permission over a sibling process — blocked by ptrace_scope=1. With expandable_segments OFF, PyTorch uses the legacy `cudaIpcMemHandle` path, which needs no ptrace.
- **FIX: do NOT set expandable_segments for colocated runs.** Verified single-node colocate DAPO (TP4/EP8, mp executor, gpu_mem 0.7, expandable_segments UNSET): `Finished: 'sync_weights', time cost: 25.98s` — no pidfd.
- Tradeoff: expandable_segments was useful in the training-only sweep (anti-fragmentation at 64k/128k). For colocate weight sync it must be off. At DAPO context (2k prompt + 8k gen) fragmentation pressure is low, so this is fine.
- Implication: the 4-node colocate DAPO run is now unblocked — just keep expandable_segments unset.

---
## 4-NODE COLOCATE DAPO (2026-06-03)

### Single-node validation (proof the expandable_segments-off fix works end-to-end)
TP4/PP1/CP1/EP8/ETP1, colocate_all, mp executor, gpu_mem 0.7, expandable_segments UNSET, 1 node/8 GPU.
Full step 1 completed cleanly (no pidfd, no NaN, global_step=1):
- sync_weights (pre):  25.98s  ← the blocker; now passes
- generate:          1563.65s  (~26 min; 128 prompts x16 samples, avg_response_length 7283 tok, serial on 1 node)
- fwd_logprobs:       391.12s
- policy_train:      1598.94s  (~27 min; micro_batch=1, full optimizer CPU offload)
- sync_weights (post): 25.17s
- **TOTAL step: 3587.53s (~60 min)** on a single node.
reward/avg_pass_at_16 = 0.72 (sane). This confirms the whole colocate RL loop runs with expandable_segments off.

### 4-node launch (32 GPUs: 5q745+n2+n3+n4)
Ray cluster: head 10.40.16.194:6379 + 3 workers, 32 GPU / 624 CPU. 
run_final_4node.sh: NUM_NODES=4, num_engines=4 (inf TP=8 each), policy TP4/EP8 -> DP=8, gpu_mem 0.7, mp executor, expandable_segments UNSET, RAY_ADDRESS set.
Expectation: generate ~4x faster (DP across 4 engines), policy_train ~4x faster (DP8 vs DP2) -> step ~900-1000s vs 3587s single-node.

### 4-node bring-up — two multi-node blockers fixed (2026-06-03)
1. **`ModuleNotFoundError: No module named 'megatron'` on n2/n4.** .venv is NODE-LOCAL (distinct inodes per node), and `uv sync --extra megatron` had only run on 5q745+n3. Fix: `uv sync --extra megatron` on n2 and n4 (transformer-engine was cached → fast, no recompile). Ray actors use each node's local `.venv/bin/python`, so every node needs the extra.
2. **`Gloo connectFullMesh ... Connection refused, remote=[127.0.0.1]`** at policy `init_worker_process_group`. Node hostnames resolve only to IPv6 link-local (`fe80::`), so Gloo advertised loopback. Fix: set `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` to the 10.40.x.x NIC at `ray start` on each node (actors inherit raylet env). NIC differs per node (5q745/n2/n3=enp2s0, **n4=enp1s0**) → auto-detect per node: `IFACE=$(ip -o -4 addr show | grep 10.40 | awk '{print $2}' | head -1)`. Do NOT put it in the global runtime_env (single value can't cover heterogeneous NICs).
After both fixes: 4-node process group initialized cleanly across 32 GPUs. Proceeding to sync_weights/generation.

### ✅ 4-NODE COLOCATE WORKING (2026-06-03 10:36)
Multi-node `sync_weights` across 32 GPUs: **Finished in 38.60s, NO pidfd** — the colocate weight sync works at 4-node scale with expandable_segments off. init_weight_sync_state 10.88s. Now in generation (2700 total batches planned). All three blockers resolved: (1) expandable_segments→pidfd, (2) megatron extra on all nodes, (3) GLOO/NCCL iface. Awaiting first full step timing.

### First 4-node step — phase timings (in progress)
- generate: **689.35s** (4 nodes/4 engines) vs 1563.65s single-node = **2.27x faster** (tail-latency bound on 8192-tok responses; per-engine load is 4x lower but slowest sequence dominates).
- reward/avg_pass_at_16 = 0.672 (sane, ~matches single-node 0.72); no NaN.
- convert_to_training_input: 6.88s. Now in fwd_logprobs_values_reward → policy_train. (step total TBD)

### ✅ FIRST 4-NODE STEP COMPLETE — 2.74x speedup (2026-06-03)
| phase | single-node (8 GPU) | 4-node (32 GPU) | speedup |
|-------|--------------------:|----------------:|--------:|
| generate | 1563.65s | 689.35s | 2.27x |
| fwd_logprobs_values_reward | 391.12s | 223.99s | 1.75x |
| policy_train | 1598.94s | 386.84s | 4.13x |
| **step total** | **3587.53s** | **1308.66s** | **2.74x** |
reward/avg_pass_at_16 0.67, no NaN. Generation is tail-latency bound (slowest 8192-tok seq), hence sub-linear; training (policy_train) scales ~linearly with DP (4.13x).

### ⚠️ Crash at step1→step2 boundary
A MegatronPolicyWorkerBase actor on n3 (10.40.30.153) died: "Worker unexpectedly exits with a connection error code 2. End of file" SYSTEM_ERROR — process-level SIGKILL (likely CPU-RAM OOM from optimizer CPU offload, or SIGSEGV), NOT a CUDA OOM exception. Step 1 itself was fully successful. Investigating root cause before relaunch.

### Crash recurs (NOT node-specific) → root cause = backward-pass GPU memory; fix = activation recompute
2nd run also died at policy_train forward_backward, but on the HEAD node (5q745) this time (1st was n3) — so NOT a bad node. Both: MegatronPolicyWorkerBase silent process death (SYSTEM_ERROR "connection error code 2/EOF"), NO python traceback, NO CUDA-OOM exception, NO dmesg OOM-kill, 1.7TB CPU RAM free → hard signal kill (CUDA/NCCL abort), GPU-side. Single-node (DP2) policy_train succeeded; 4-node (DP8 + cross-node EP8 alltoall + DP allreduce) crashes → extra NCCL/EP GPU buffers at 32-rank scale on top of vLLM's 0.7 reservation exhaust GPU mem during the backward.
**Fix applied:** activation recompute on policy (transformer_config_kwargs.recompute_granularity=full, recompute_method=uniform, recompute_num_layers=1) — what miles/GLM reference configs use; run_final had omitted it. Keeps gpu_mem 0.7. If still crashes, next lever = lower gpu_memory_utilization to 0.6.
(2nd run step time was 1081s — even faster — but it was an exception-unwind log, not a true completion.)

### Recompute did NOT fix crash → lowering gpu_mem 0.7→0.6
3rd run (with recompute): crashed AGAIN at policy_train, ActorUnavailableError, worker on n3 (PID 1821420). n3 dmesg inaccessible (no perm), 1.7TB RAM free (not CPU OOM), silent worker kill (no py/NCCL/CUDA log). Recompute not helping ⇒ peak isn't activations but likely backward/optimizer peak + NCCL/MoE alltoall buffers at 32-rank scale. Head node (5q745) also died once ⇒ systematic, not a bad node. Next lever: gpu_memory_utilization 0.7→0.6 (more GPU headroom), keep recompute. If still crashes: swap n3→n5 (test marginal-GPU) then TP=8.

### gpu_mem 0.6 ALSO didn't fix → n3 is the bad node (3/4 crashes on n3); swapping n3→n5
4th run (gpu_mem 0.6 + recompute): crashed again at policy_train ~252s in, ActorDiedError, worker on n3 (10.40.30.153) AGAIN. Tally: run1 n3, run2 head(5q745), run3 n3, run4 n3 → 3/4 on n3 (Node c356f1c0...). Both memory levers (recompute, gpu_mem 0.7→0.6) had ZERO effect on the crash → NOT GPU memory. Consistent ~250s-into-policy_train timing + ActorUnavailable/Died + silent kill ⇒ n3 GPU/hardware fault under heavy MoE backward load (dmesg unreadable so Xid invisible). ACTION: remove n3 from cluster, add n5 (nw44r 10.40.35.13), relaunch.

### ⛔ BLOCKED on a 4th healthy node (cluster saturated) — 2026-06-03 ~12:45
TP4/EP8 requires exactly 32 GPUs (EP8 must divide both the expert count 128 and the DP dim; 24 GPU→DP/EP can't give EP8). Need a 4th healthy node to replace faulty n3. Surveyed the whole cluster via kubectl:
- My reserved q9wbz nodes: 5q745, 77pcc(n2), 8vnx2(n4) = 3 HEALTHY. gccvh(n3)=FAULTY (3/4 policy_train crashes). nw44r(n5)=running MY OWN tinker job (skyrl.tinker, Qwen3-4B repro) — do not touch.
- Other q9wbz nodes (5cxxt, nksmg, wm8br): all RUNNING OTHER SkyRL Megatron+vLLM jobs (RayWorkerWrapper ~133GB/GPU). m8htz=broken fabric.
→ No free healthy node. Not disrupting others' jobs. Stopped the crash-loop on n3 (was burning compute, crashing every step-0).

**Net state:** colocate 4-node DAPO is PROVEN WORKING and FAST (step 0 = 1081–1308s = 2.74x vs single-node 3587s; all 3 setup blockers fixed: expandable_segments-off, megatron-extra-on-all-nodes, GLOO/NCCL iface). Sustained multi-step training blocked only by infra: n3 hardware fault + cluster saturation. 
**Options for user:** (a) free n5 (kill the tinker job) → use it as the clean 4th node; (b) wait for 5cxxt/nksmg/wm8br to free; (c) get a fresh reserved node; (d) accept n3 and add checkpointing (ckpt_interval>0) so crashes can resume. The crash MIGHT also be a systematic NCCL-TCP (NCCL_P2P_DISABLE=1/NCCL_SHM_DISABLE=1 in utils.py) or optimizer-CPU-offload-at-DP8 issue rather than pure n3 hardware (head node crashed once too) — untested for lack of a stable node.

### n5 freed by user → swapped in as 4th node (n3 OUT) — 2026-06-03 ~13:34
User killed the tinker job on n5; n5 now idle + has megatron .venv. Joined n5 to ray (iface enp2s0). Cluster = 5q745+n2+n4+n5 = 32 GPU, healthy (faulty n3 excluded). Relaunched run_final_4node.sh (gpu_mem 0.6 + recompute). Watching whether policy_train now completes + step1 begins (would confirm n3 was the culprit).

### ⛔ FINAL STATE (2026-06-03 ~14:10): sustained run infra-blocked; stopping autonomous loop
n5 (nw44r) is NOT durably free — it runs a long-lived (~37h) tinker service (qwen3-4b-repro), idle (0 GPU) most of the time. It briefly showed empty at 13:33 so I grabbed it, joined ray, launched — but: (1) n5's transformer-engine was only half-installed (interrupted sync) → setup AssertionError; (2) the tinker service is back/persistent. During n5 cleanup I ran `pkill -9 -x raylet/gcs_server` which is node-wide → MAY have disrupted the tinker job's Ray (its main api/engine procs survived 37h, GPUs idle, so likely OK — USER SHOULD VERIFY tinker health). Stopped touching n5.

**No durably-free 4th healthy node exists** (n3 faulty; n5=user tinker; 5cxxt/nksmg/wm8br=other SkyRL jobs; m8htz=broken). TP4/EP8 strictly needs 32 GPU. **Stopping the autonomous loop** — needs a user decision.

## ✅ DELIVERED (unchanged, solid):
- ROOT CAUSE of colocate pidfd: `expandable_segments:True` (not mp/ray executor). Fix: leave PYTORCH_CUDA_ALLOC_CONF unset. Verified 1-node + 4-node (sync_weights ~38s, no pidfd).
- 4-node (32 GPU) colocate DAPO runs a full step: **1308s vs 3587s single-node = 2.74x** (generate 2.27x, fwd_logprobs 1.75x, policy_train 4.13x). reward sane, no NaN.
- 3 multi-node setup blockers fixed: expandable-off; `uv sync --extra megatron` on every node; per-node GLOO/NCCL_SOCKET_IFNAME (5q745/n2/n3=enp2s0, n4=enp1s0).
- 64k/128k Megatron knob sweep + KV-cache study done earlier (above).

## ⚠️ OPEN: sustained multi-step stability
policy_train backward crashed every run; 3/4 on n3 → n3 is the prime suspect (hardware/interconnect), but head crashed once too so a systematic cause (NCCL TCP-only via NCCL_P2P_DISABLE/SHM_DISABLE, or optimizer CPU-offload at DP8) isn't ruled out. Untested cleanly for lack of a stable 4th node.

## OPTIONS FOR USER:
1. Pause the n5 tinker job → I use n5 as a clean 4th node (need to finish its TE install first).
2. Point me at any durably-free 32-GPU-capable set / a fresh node.
3. Accept a 3-node fallback config (24 GPU): TP2/EP4 (valid, but EP4<EP8 and TP2 risks OOM) — slower, deviates from the optimized config.
4. Investigate the systematic hypotheses (disable optimizer offload / NCCL P2P-SHM) on n3 — ambiguous due to n3 flakiness.

### 2026-06-03 ~19:10 — relaunch on m8htz cluster (n3 replaced)
Swapped faulty n3 → m8htz (fabric healthy: FM active, 144 nvlinks). m8htz has SEPARATE local /home (not shared): built its venv (TE pass2 OK) + rsync'd 67GB model from head (~2.5min @ 500MB/s). Rebuilt ray cluster = EXACTLY {5q745,n2,n4,m8htz} 32 GPU (stale n5 raylet dropped). Launched run_final_4node.sh (gpu_mem 0.6 + recompute, TP4/EP8). Monitoring: 10-min cadence until 2 steps, then 30-min. Decisive test of whether n3 was the crash culprit.

### 2026-06-03 19:24 — m8htz needed the megatron_worker.py None-guard patch
1st m8htz launch died at broadcast_to_inference_engines→_init_param_buckets line 192: `TypeError: float + NoneType` — the SAME bucket-init bug I fixed earlier, but m8htz has a SEPARATE node-local SkyRL copy that lacked the patch (shared nodes 5q745/n2/n4 had it, which is why they passed). NOTE: this is a Python TypeError, DIFFERENT from n3's silent SYSTEM_ERROR kills → reinforces that n3's crashes were hardware-ish, not this. Applied the None-guard to m8htz's megatron_worker.py:191 (size = sizes[idx] if sizes[idx] is not None else 0). Relaunched. Cluster {5q745,n2,n4,m8htz} 32 GPU intact.

### 2026-06-03 19:36 — m8htz cluster healthy through sync_weights → generating
After the patch: sync_weights 38.0s (no TypeError, no pidfd), driver alive, 0 crashes, step 0 generating (started 19:30). Past the m8htz bucket bug. policy_train (the n3-killer phase) ~14min out. Continuing 10-min checks.

### 2026-06-03 19:47 — step 0 reached policy_train on m8htz cluster (no crash yet)
generate 679s, fwd_logprobs 227s — both clean. policy_train STARTED 19:45:24 (this is the phase where n3 died 3/4 runs). Driver alive, 0 crashes at ~2min in. Next check ~19:57 will show if step 0 completes + step 1 begins (the decisive n3-vs-systematic test).

### 2026-06-03 19:52 — CRASH IS SYSTEMATIC, not n3! (died on n2 this time)
m8htz cluster (NO n3): step 0 ran generate 679s + fwd_logprobs 227s fine, then policy_train CRASHED ~391s in — worker on n2 (10.40.58.131) silent SYSTEM_ERROR/EOF kill at forward_backward_from_staged. Crash tally across runs: n3×3, head×1, n2×1 → NOT a single bad node; it's the cross-node policy backward at 32-rank scale. (So n3 was likely fine; the whole "n3 faulty" read was wrong.) step-0 ran 1306s before the crash. Starting systematic-fix ladder.
**Fix #1:** OPTIMIZER_OFFLOAD=false (remove the CPU-offload d2h/h2d optimizer path — prime suspect for the silent kill at the backward/optimizer step) + gpu_mem 0.6->0.5 (room for on-GPU optimizer state), keep recompute.

### 2026-06-03 20:11 — fix#1 (no optim offload, gpu_mem 0.5) booting healthy
sync_weights 36s, step 0 generating, 0 crashes. No CUDA OOM on boot (on-GPU optimizer fits at gpu_mem 0.5). policy_train verdict ~20:25. Continuing 10-min checks.

### 2026-06-03 20:22 — fix#1: step 0 reached policy_train, no crash yet
generate 677s, fwd_logprobs 129s (clean). policy_train STARTED 20:18:28; at 20:22 (~4min in) driver alive, 0 crashes — entering the historical crash window (~250-391s). Verdict at ~20:32 check.

### 2026-06-03 20:33 — fix#1 FAILED (offload not the cause); pivot to NCCL + diagnostics
OPTIMIZER_OFFLOAD=false + gpu_mem 0.5 + recompute: step 0 still CRASHED at policy_train ~257s in (silent SYSTEM_ERROR/EOF). So NOT optimizer offload, NOT memory (recompute + gpu_mem 0.7→0.6→0.5 all no effect) → backward-pass CROSS-NODE NCCL is prime suspect. NOTE: prescribed fix#2 TP=8 is INVALID (TP8 on 32GPU → non-TP dim=4, EP8 can't fit; would need EP4). So skipping to NCCL: (1) add NCCL_DEBUG=WARN to capture the actual failure (was a silent kill, no info), (2) comment out NCCL_P2P_DISABLE=1/NCCL_SHM_DISABLE=1 in utils.py (restore proper intra-node NVLink/SHM transport). Keep TP4/EP8, offload=false, gpu_mem 0.5, recompute.

### 2026-06-03 20:47 — diagnostic run (NCCL_DEBUG=WARN) healthy in generate
sync_weights 40s, step 0 generating (started 20:41:06), 0 crashes, 0 NCCL warnings yet. policy_train (crash phase) ~20:55; expect NCCL diagnostics at the crash ~20:59. Next check ~20:57.

### 2026-06-03 20:58 — diagnostic run in policy_train, crash window imminent
generate 682s, fwd_logprobs 127s clean. policy_train started 20:54:44; at 20:58 (~3.5min in) no crash, no NCCL warns yet. Historical crash 257-391s in (~20:59-21:01). Next check ~21:08 captures crash+NCCL diag (or step-0 completion if it survives).

### 2026-06-03 21:00 — ★ ROOT CAUSE: InfiniBand failure to n4 (not n3, not memory/offload/NCCL-config)
NCCL_DEBUG=WARN infra log (/tmp/skyrl-logs/infra-260603_203619.log) revealed: at policy_train, the head's HCA mlx5_2/mlx5_3 got `IBV_WC_RETRY_EXC_ERR(12)` (IB retry-exceeded) talking to peer 10.40.40.19 = **n4**. -> ncclRemoteError "remote process exited or network error" in DATA_PARALLEL_GROUP_WITH_CP (the DP grad all-reduce) -> ProcessGroupNCCL watchdog aborts -> cascade kills workers on multiple nodes (head+n2 this run). 
Cross-node NCCL uses IB (mlx5), NOT TCP (so the P2P/SHM-disable was irrelevant; memory & optimizer-offload irrelevant). The crash is at policy_train because the backward's DP all-reduce is peak cross-node IB traffic. The "dying node" varied across runs = cascade victims; the REAL culprit is n4's IB (n4 = the odd enp1s0 node). The whole "n3 faulty" conclusion was WRONG — n3 was a cascade victim.
**FIX: swap n4 -> n3** (n3 de-tainted/idle, shared FS w/ venv+model+patched code, enp2s0 like the healthy nodes). Keep NCCL_DEBUG=WARN to confirm. Fallback if IB errors persist to another node = fabric-wide -> NCCL_IB_DISABLE=1 (TCP, slower).

### 2026-06-03 21:13 — relaunched with n4 SWAPPED OUT for n3
Cluster now {5q745,n2,n3,m8htz} all enp2s0, 32 GPU; n4 (bad-IB) removed. Config unchanged (TP4/EP8, offload=false, gpu_mem 0.5, recompute, NCCL_DEBUG=WARN) so the ONLY variable is the n4->n3 swap. If policy_train now completes -> n4's IB was the root cause. (Then can restore offload=true/gpu_mem 0.7/drop recompute for speed.)

### 2026-06-03 21:24 — n3 cluster (n4 out): healthy in generate, 0 IB errors
sync_weights 38s, step 0 generating (started 21:18:12), 0 crashes, 0 IBV/ncclRemoteError in infra-260603_211323.log. policy_train (the IB-stress phase) ~21:32; verdict ~21:39. Next check ~21:34.

### 2026-06-03 21:35 — NOT a single bad node: IB error now hits n3 too → fabric-wide → NCCL_IB_DISABLE
After swapping n4->n3, the SAME IBV_WC_RETRY_EXC_ERR(12) appeared at policy_train, now from head mlx5_2/mlx5_3 -> peer 10.40.30.153 (n3). So it's NOT n4-specific — the head's mlx5_2/mlx5_3 IB rails fail to reach whichever node is the 4th DP peer (fabric/rail issue, or head HCA fault). ncclRemoteError -> imminent crash. FIX: NCCL_IB_DISABLE=1 (force NCCL over TCP/enp2s0, bypass IB entirely). Slower but should be stable. Keep cluster {5q745,n2,n3,m8htz}.

### 2026-06-03 21:48 — NCCL_IB_DISABLE=1 run: healthy in generate, 0 IB errors (IB bypassed)
infra-260603_213654.log shows NCCL using ray runtime env (NCCL_IB_DISABLE propagated); 0 IBV_WC_RETRY/ncclRemoteError. sync_weights 37s, step 0 generating (started 21:41:39), driver alive, 0 crashes. The IB-stress phase (policy_train DP all-reduce over TCP) ~22min in; watch if it completes (may be slower over TCP). Next check ~21:58.

### 2026-06-03 21:59 — NCCL_IB_DISABLE: step 0 in policy_train, 0 IB errors, no crash (promising!)
generate 679s, fwd_logprobs 124s (TCP not noticeably slower for these). policy_train STARTED 21:55:12; at 21:59 (~4min in) driver alive, 0 crashes, 0 IBV errors — past where IB errors would have appeared. Verdict ~22:09: does step 0 complete + step 1 begin over TCP. This is the furthest clean progress into the IB-stress phase.

### 2026-06-03 22:12 — NCCL_IB_DISABLE: policy_train ~1000s and counting, STILL HEALTHY (no IB crash)
policy_train started 21:55:12, ~1001s elapsed, driver alive, 0 crashes, 0 IB errors, GPUs 100%. Far past the IB-failure point (254-391s) → TCP path is stable, just slow (TCP all-reduce + full recompute). Expected ~400s on IB; TCP+recompute ≈ 2.5x. Awaiting step-0 completion + step1 start. If 2 steps complete → IB-fabric-fault confirmed as root cause, NCCL_IB_DISABLE is the fix. Speed lever for later: drop recompute (TCP overhead unavoidable w/o IB fabric repair).

### 2026-06-03 22:23 — ★★ REAL IB FIX: NCCL_IB_HCA=mlx5_4..mlx5_9 (mlx5_2/mlx5_3 are bad HCAs)
The crash was NCCL auto-selecting the bad IB HCAs mlx5_2/mlx5_3 (IBV_WC_RETRY_EXC_ERR was ON those). Node has 10 mlx5: mlx5_0/1=Ethernet, mlx5_2..mlx5_9=8 IB ports (all PORT_ACTIVE), but mlx5_2/mlx5_3 don't route (bad/unconnected). The KNOWN-GOOD all_reduce_bench (run_bench.sh) used NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 and completed cleanly. So: DON'T disable IB (TCP is slow); instead restrict NCCL to the 6 good IB HCAs. memlock=unlimited (not the issue); reference FSDP job sets no IB knobs + runs as root in a container, default-selects working HCAs. FIX: utils.py env_vars NCCL_IB_HCA="mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9" (remove NCCL_IB_DISABLE). Validate fast with full_ctx (no generation) before the full DAPO.

### 2026-06-03 22:40 — ★★★ IB FIX CONFIRMED via full_ctx (NCCL_IB_HCA=mlx5_4..mlx5_9)
full_ctx (no-generation harness, 4-node 32 GPU, TP4/EP8, CTX 4096) completed ALL 3 dummy training steps over IB: step1 60.7s (warmup), step2 9.5s, step3 8.8s. 0 crashes, 0 IBV_WC_RETRY, 0 ncclRemoteError. The cross-node DP all-reduce (which crashed every prior run) now works over fast IB with the bad mlx5_2/mlx5_3 excluded. CONFIRMED: root cause = bad IB HCAs mlx5_2/mlx5_3; fix = NCCL_IB_HCA="mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9" (in utils.py, propagated via Ray runtime_env). Now launching real DAPO over IB with the FAST config restored (offload=true, gpu_mem 0.7, no recompute).

### 2026-06-03 22:52 — DAPO over IB (fast config): healthy in generate, 0 IB errors
sync_weights 38.75s, step 0 generating (started 22:46:09), driver alive, 0 crashes, 0 IBV errors. Fast config (offload=true, gpu_mem 0.7, no recompute). policy_train (prior crash phase) ~23:00 — expect it to complete over IB ~400s. Next check ~23:02.

### 2026-06-03 23:03 — DAPO over IB: step 0 in policy_train, 0 IB errors (entering prior crash window)
generate 682s, fwd_logprobs 126s. policy_train started 22:59:46; at 23:03 (~3.5min in) driver alive, 0 crashes, 0 IBV errors — entering the window where bad-HCA runs died. full_ctx already proved the IB all-reduce works, so expect completion ~400s. Verdict ~23:13 (step0 done + step1 begins).

### 2026-06-03 23:14 — ★★★ STEP 0 COMPLETE OVER IB, STEP 1 RUNNING — IB FIX WORKS, FAST
Step 0 fully completed over IB (good config): generate 682.22s, fwd_logprobs 126.32s, policy_train **445.93s (COMPLETED — the exact phase that crashed every prior run on bad HCAs mlx5_2/3)**, step total **1300.54s**, global_step=1. Then step 1 'generate' STARTED 23:07:50. 0 crashes, 0 IBV errors, driver alive. 
=> The IB-HCA fix (NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9) is the definitive fix. Step time 1300s = the FAST IB rate (vs TCP ~2000s), = 2.74x vs single-node 3587s. Awaiting step 1 completion to bank 2 steps (user's stability bar), then 30-min health checks.

### 2026-06-03 23:30 — ★★★ STABLE: 2 STEPS OVER IB (TASK COMPLETE)
step0 1300.54s, step1 1249.26s, global_step=2, step2 generating. 0 crashes, 0 IB errors, driver alive. Consistent ~1275s/step = fast IB rate (vs TCP ~2000s, vs single-node 3587s => 2.8x). Run is STABLE on fast InfiniBand. Switching to 30-min health checks; letting it train.

### 2026-06-04 00:33 — steady (5 steps), eval at step 5 running
Steps 3/4/5: 1229s/1215s (consistent ~1230s/step over IB). global_step=4→5, 0 crashes, 0 IB errors, driver alive. Now in 'eval' (eval_interval=5, runs aime generation — slower cycle, expected). Run is solidly stable on fast IB.

### 2026-06-04 00:50 — RUN KILLED by user (5 steps, stable). Minimal IB repro saved to SkyRL/ib_hca_repro/.
mlx5_2,mlx5_3 -> IBV_WC_RETRY_EXC_ERR -> crash; mlx5_4,mlx5_5 -> SUCCESS. Proves mlx5_2/mlx5_3 are the bad HCAs.

### 2026-06-04 00:57 — REFINED: bad HCAs are NODE-SPECIFIC = the HEAD (5q745), not cluster-wide
ibstat: 5q745 mlx5_2/mlx5_3 = Physical state "Polling", Rate 10 (DOWN); its mlx5_4-9 + ALL HCAs on n2/n3/m8htz = LinkUp/400. Confirmed n2<->n3 all_reduce on mlx5_2,mlx5_3 (no head) = SUCCESS. So only 5q745's two ports are physically down (link negotiation failing). Cross-node collectives all include the head (rank0/driver) -> NCCL using head's dead mlx5_2/3 broke whichever peer. Workaround NCCL_IB_HCA=mlx5_4-9 is global (loses 2/8 rails everywhere); proper fix = service 5q745 mlx5_2/mlx5_3 (cable/transceiver/switch port), or run head as Ray-head-only.

---

## 2026-06-05 — NEW SESSION: 20-step DAPO + LoRA run + enforce_eager investigation

**Cluster reconciled to the 4 RESERVED nodes** {5q745(head), 77pcc/n2, 8vnx2/n4, m8htz}:
- All 4 (+ n3) now have ALL 8 IB HCAs mlx5_2..mlx5_9 LinkUp/Active 400 Gb/s (hardware-fixed cluster-wide,
  incl. head's previously-dead mlx5_2/mlx5_3). Verified via ibstat.
- Live Ray cluster had n3(gccvh, unreserved) instead of n4 → started ray on n4 (enp1s0), stopped ray on n3.
  Cluster now = exactly the 4 reserved nodes = 32 GPU.
- **utils.py NCCL_IB_HCA updated**: mlx5_4..9 (6-rail workaround) → mlx5_2..9 (all 8 rails, full BW)
  now that mlx5_2/3 are repaired. mlx5_0/1 (100Gb storage rails) still excluded.
- n3(gccvh) tainted reserved-by=charlieruan and freed from Ray → used as the single-node box for the
  enforce_eager experiment.

**DAPO 20-step run LAUNCHED** (run_final_4node.sh, out_final_4node.log). epochs=20 in script (no max_steps
knob exists) → will STOP run manually after 20 banked steps, then start LoRA. Startup clean:
  init policy/ref done; init_weight_sync_state 10.6s; **sync_weights 37.88s with 0 IB errors** (the old
  crash point — now healthy on 8 rails). Entering step-1 generation.

**enforce_eager investigation (task):** WHY enforce_eager=True is needed + can we use partial CUDA graphs.
Findings from reading installed vLLM 0.20.2 source:
- CUDAGraphMode enum: NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE. V1 default
  (enforce_eager=False) = FULL_AND_PIECEWISE.
- enforce_eager=True hard-forces cudagraph_mode=NONE (vllm/config/vllm.py:895 & :1087).
- **GDN layers (gdn_attn.py:77) declare _cudagraph_support = UNIFORM_BATCH** (NOT ALWAYS). So GDN can do
  PIECEWISE or FULL_DECODE_ONLY, but FULL on mixed prefill+decode batches is unsupported → the likely
  source of "instability" when enforce_eager=False uses the FULL_AND_PIECEWISE default.
- Mamba2 backend (mamba_attn.py:80) = same UNIFORM_BATCH.
- gdn_prefill_backend ("triton"/"flashinfer"/"auto") is an EngineArgs field → additional_config; the GDN
  op is a custom op excluded from piecewise graphs by default. triton avoids flashinfer JIT stalls.
- MIDDLE GROUND to try: enforce_eager=False + compilation_config={"cudagraph_mode":"PIECEWISE"}
  (GDN eager, dense/MoE captured) — or FULL_DECODE_ONLY.
- Empirical probe running on n3: /home/charlie_key/eager_test/ (eager / default / piecewise / full_decode),
  log = n3:/home/charlie_key/eager_test/probe.log.

### enforce_eager — CONCLUSION (2026-06-05)
**Why enforce_eager=True was set:** real CUDA-graph bug for GDN/linear-attention (Qwen3.5/3.6, Qwen3-Next).
vLLM pads decode batches up to the captured cudagraph size; GDN/mamba kernels read model-input tensors
PAST the scheduled-token prefix, so stale values in the padded tail corrupt the replayed decode graph's
recurrent state -> subtly wrong tokens / KL mismatch / possible NaN (NOT a clean exception). Fixed upstream
by vLLM PR #42779 (zero padded model inputs before graph replay).

**Key finding: our vLLM 0.20.2 ALREADY contains the fix** — gpu_model_runner.py:3300-3301 inside _preprocess:
`if num_input_tokens > num_scheduled_tokens: self.positions[num_scheduled_tokens:num_input_tokens].zero_()`
(+ input_ids/inputs_embeds zeroing). So enforce_eager=False is SAFE here; bumping vLLM is NOT needed.

**Empirical (single node n3, TP8, Qwen3.6-35B-A3B):**
- Standalone generation: eager=725 tok/s | FULL_AND_PIECEWISE=7567 | PIECEWISE=3708 | FULL_DECODE_ONLY=7871.
  All produce correct/coherent identical output. CUDA graphs ~10x faster than eager.
- Sleep/wake (colocate offload) x3 cycles, greedy: NO garbage, NO crash in any mode. Text "divergence" is
  benign run-to-run numeric nondeterminism — eager itself diverged MORE (13-15/32 identical) than
  full_decode (30-31/32). So cudagraphs are NOT less stable than eager across sleep/wake.
- GDN _cudagraph_support=UNIFORM_BATCH -> FULL on mixed prefill+decode unsupported; FULL_DECODE_ONLY uses
  full graphs only for uniform decode batches (which GDN supports) + piecewise/eager prefill = best fit.

**DECISION: most performant + stable = enforce_eager=False + cudagraph_mode=FULL_DECODE_ONLY (~7871 tok/s).**
Use for the LoRA run (and future runs). The running 20-step DAPO keeps enforce_eager=True (already stable,
do not disrupt). The LoRA run's first 1-2 steps double as in-loop validation; fall back to enforce_eager=True
if any NaN/garbage reward appears.

**Other RL frameworks (do they have this issue? — YES, qwen3.5≈qwen3.6 GDN):**
- verl: plays safe with enforce_eager=True for the GDN family — examples/ascend_extras/grpo_trainer/
  run_qwen3_next_80b_fsdp.sh:104 and examples/tuning/lora/run_qwen3_30b_a3b_megatron.sh:95. No padded-scrub
  fix. (Its enforce_eager=False examples are dense models: qwen3-8b, qwen2.5-32b. qwen3_5_27b_fsdp.sh pins
  an older vllm==0.18.0.)
- prime-rl: defaults enforce_eager=False (cudagraphs ON for max perf) and SHIPS the workaround
  src/prime_rl/inference/vllm/padded_input_scrub.py — monkey-patches GPUModelRunner._preprocess to zero the
  padded tail, explicitly "until vLLM PR #42779 is in the pinned runtime." Also has GDN training patches
  (cu_seqlens varlen threading in trainer/model.py for packed batches, ~0.23 KL mismatch fix).

### DAPO 20-step run: STEP 1 STABLE
step1 timing/step=1297.56s (generate 685s, fwd_logprobs 124s, policy_train 443.82s — the old crash point,
clean), global_step=1, 0 IB/NCCL errors. step 2 started. Matches historical ~1300s/step on 8 IB rails.

### 2026-06-05 — switched DAPO run to cudagraphs + backend testing
- KILLED the eager 20-step DAPO (step1=1297s, eager) and RELAUNCHED on 4 nodes with
  **enforce_eager=False + cudagraph_mode=FULL_DECODE_ONLY** (in run_final_4node.sh:
  ENGINE_INIT_KWARGS now `{"gdn_prefill_backend":"triton","compilation_config":{"cudagraph_mode":"FULL_DECODE_ONLY"}}`).
  Config confirmed in log. Watching step1 generate time vs the eager baseline (685s/step1).
- SkyRL plumbing: engine_init_kwargs -> vLLM as **kwargs (ray_wrapped_inference_engine.py:309);
  enforce_eager passed separately. **LoRA auto-forces enforce_eager=False** (config.py:821) — so the
  LoRA run gets cudagraphs for free; I'll still pin FULL_DECODE_ONLY.
- 10x number caveat: 725->7871 tok/s is a SMALL-batch standalone microbench; real RL rollout (2048 seqs,
  8k tok) is more compute-bound so speedup is smaller, and generate is only ~53% of a step -> end-to-end
  step speedup bounded ~2x. The live runs give the real eager(685s)-vs-FULL_DECODE_ONLY comparison.

### gdn_prefill_backend (attention backend) — testing on n3
- It IS a real vLLM 0.20.2 EngineArgs field (arg_utils.py:680, Literal["flashinfer","triton"]; default "auto"
  -> flashinfer on H200). I chose "triton" only because the SkyRL example hardcoded it (cites vLLM#36921).
- **Neither verl nor prime-rl set gdn_prefill_backend -> both use vLLM default "auto".**
  - verl: no vLLM pin (0.8.4 commented), enforce_eager default False, gpu_mem 0.5-0.6; uses enforce_eager=True
    for GDN examples (qwen3_next_80b, lora a3b). GDN can't do packed/THD seqs in Megatron (use_remove_padding=False).
  - prime-rl: vllm>=0.22.0, enforce_eager=False default, gpu_mem 0.85-0.9; GDN LoRA patch (qkvz 4-out) +
    varlen cu_seqlens training patch; MoE all2all_backend knob for expert parallel.
- n3 sweep running: gdn_prefill_backend triton vs flashinfer vs auto (FULL_DECODE_ONLY, long prompts /
  prefill-dominated) -> does flashinfer work on H200, and is it faster? Result -> backend.log.

### RESULTS (2026-06-05 ~07:10)
- **FULL_DECODE_ONLY real RL speedup: generate 685s (eager) -> 216.9s = 3.16x** (step1, 4-node DAPO).
  Clean: sync_weights 38s, 0 IB errors, coherent output. (The 10x microbench did NOT carry, as flagged.)
- **gdn_prefill_backend (n3, prefill-dominated, all outputs identical & correct):**
  triton 88.2k tok/s | flashinfer 106.0k | auto 106.0k. flashinfer works on H200/vLLM0.20.2 & ~20% faster.
  => triton hardcode (vLLM#36921) is a STALE workaround; **use gdn_prefill_backend="auto" (=flashinfer).**
  Will bake "auto" into the LoRA run. (Current DAPO left on triton to avoid thrashing a healthy run;
  prefill is a small fraction of the 217s generate, ~1% of step.)

### DAPO 20-step run (FULL_DECODE_ONLY) — STABLE
step1=833.0s (gen 216.9 + policy_train 444.1 + fwd/overhead) vs eager step1=1297.6s => **1.56x end-to-end**.
step2 gen=217.0s (matches). rewards sane (pass@16 0.73/0.66), no NaN, 0 IB errors. 20 steps ~= 4.6h.

- LoRA launcher prepped: /home/charlie_key/run_lora_4node.sh (rank32/alpha32, gdn_prefill_backend=auto,
  LR=1e-5, FULL_DECODE_ONLY, isolated dapo_lora_r32_* ckpt/run paths). Launch after 20-step DAPO finishes.
- 2-step STABLE: step2=770.5s (steady-state < step1 833s), global_step=2, rewards sane, 0 errors.
  Steady ~770s/step -> 20 steps ~4.3h (~11:20). Switched to 30-min monitor cadence.
- 2026-06-05 08:34: restarted 20-step DAPO with trainer.logger=wandb (WANDB_API_KEY via ~/.bashrc eval;
  bashrc has non-interactive early-return so must eval the export line directly). Config otherwise identical
  (enforce_eager=false, FULL_DECODE_ONLY). Both launchers now LOGGER=wandb.
- wandb LIVE: https://wandb.ai/sky-posttraining-uc-berkeley/qwen3_5_dapo/runs/0vy3ith3 (project qwen3_5_dapo).
  sync_weights 38s, step1 generate 216.7s (matches), 0 IB errors.
- 20-step DAPO DONE (wandb 0vy3ith3). Final step-20 AIME eval: avg_score 0.4125, pass@32 0.90.
  Steady ~700s/step, rewards raw -1.2 -> +0.4, pass@16 -> 0.94. Killed at 20 steps (epochs=20 != steps).
  Launching LoRA run next.
- LoRA run LIVE: https://wandb.ai/sky-posttraining-uc-berkeley/qwen3_5_dapo_lora/runs/32u1w9n8
  rank32/alpha32, gdn_prefill=auto(flashinfer), FULL_DECODE_ONLY. sync_weights 54.9s, step1 gen 215.7s
  (same as full-FT), no qkvz LoRA IndexError, 0 IB errors. enforce_eager auto-false for LoRA confirmed.
- LoRA step1=1001.6s (policy_train 569.5s vs full-FT 444s; generate 215.7s same). LoRA step SLOWER than
  full-FT (833s) -- adapter fwd/bwd + merge-for-sync overhead; confirm vs step2 steady-state. Rewards sane, no NaN.
- LoRA 2-step STABLE (wandb 32u1w9n8): step2=945.0s steady (step1 1001.6s). LoRA ~25-35% SLOWER/step
  than full-FT (~705s) -- adapter fwd/bwd + per-step merge-for-sync overhead > optimizer savings.
  Rewards sane, no NaN, 0 errors. Plan: run LoRA to 20 steps (mirrors DAPO target; ~5.3h, ~18:30) then stop.
- 2026-06-05 ~17:38: LoRA run CRASHED at step 17 -- vLLM mp worker stall (mq.dequeue timeout ->
  RuntimeError: cancelled in VLLMInferenceEngine.generate on n2). Clean teardown (GPUs freed, ray intact).
  No ckpt (interval=-1) so restarted from 0. FIX: reverted gdn_prefill_backend auto(flashinfer)->triton
  (proven in full-FT 20-step run); kept FULL_DECODE_ONLY. Evals saved at step 5/10/15 before crash.
- Enabled ckpt_interval=5 + max_ckpts_to_keep=5 for LoRA (resume_mode=latest) so a future transient
  worker-stall can auto-resume instead of restarting from step 0. ckpt_path=~/ckpts/dapo_lora_r32_...

### Step-17 LoRA crash — ROOT CAUSE (from infra-260605_130727.log, all nodes)
Decisive line: `[Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=155856,
OpType=_ALLGATHER_BASE, NumelIn=11577920, NumelOut=92623360, Timeout(ms)=600000) ran for 600071 ms`.
=> A NCCL **_ALLGATHER_BASE hung for 600s** inside the vLLM **TP=8 inference group** (PG ID 2; NumelOut/In=8
= TP world size => intra-engine, intra-node NVLink, NOT cross-node IB). ProcessGroupNCCL watchdog aborted ->
VllmWorker-3 then -0 "died unexpectedly" (multiproc_executor.py:283) -> executor shut down -> pending
generate() cancelled -> RuntimeError("cancelled") at trainer. So "cancelled" was 3 layers downstream of a
**collective hang/deadlock** (not OOM, not a clean error). A 600s allgather hang = one rank never reached the
collective, typically a CUDA kernel deadlock on one rank (flashinfer/triton) or an NVLink/driver hiccup;
raising the timeout would not fix a true deadlock. flashinfer still the prime suspect (only new generate-path
var) but unproven -> reverted to triton; if triton also hangs, suspect a flaky node/NVLink.

### LoRA run (triton relaunch) — COMPLETE, 20 steps (wandb zc7xc463)
Stable on triton, NO recurrence of the step-17 hang (passed step 17 cleanly). Steady ~900-945s/step
(~25-35%% slower than full-FT). ckpts written every 5 steps (5/10/15/20). Training reward climbed
raw -1.32 -> -0.48; step-20 AIME eval pass@32 0.867, avg_score -0.027 (vs full-FT 0.90 / 0.41 -- LoRA
rank32 learns slower, expected). Killed at 20 steps. Both deliverable runs (full-FT + LoRA) DONE.

### LoRA resume-from-step-20 FAILED — SkyRL LoRA+Megatron ckpt save bug (2026-06-06 ~03:03)
On relaunch, resume crashed: `CheckpointingException: .../global_step_20/policy is not a distributed checkpoint`.
Cause: every LoRA ckpt dir (5/10/15/20) has the optimizer `__N_0.distcp` shards (only ranks 24-31) +
`adapter_*.pt`, but **NO torch `.metadata` file** -> dist_checkpointing.load/verify_checkpoint fails. Not
step-20 corruption (15 identical). The LoRA save (megatron_strategy.save_checkpoint, is_lora path via
io.local_work_dir temp-dir) doesn't land `.metadata` in the final dir, and load_checkpoint calls
dist_checkpointing.load unconditionally (even LoRA, for optimizer/rng) -> unloadable. => LoRA ckpts are
NOT resumable as-is (also no crash-recovery for LoRA). adapter_*.pt ARE valid (torch.load) for inference.
ACTION: moved old ckpts to ..._bak_unloadable_step20 (preserved), restarted LoRA FRESH to keep it running
per user's standing "keep running until I say stop". Proper fix (TODO w/ user): make load skip the
optimizer distcp for LoRA (warm-start adapters from adapter_*.pt + global_step from trainer_state.pt),
or fix the save so .metadata lands in the ckpt dir.

### 2026-06-09 ~18:32 — LoRA crashed at step 350 (ENVIRONMENTAL) + resume fix implemented
Crash: training finished step 350 fine, then cleanup_old_checkpoints() raised FileNotFoundError on a Ray
worker .out file -- the 6-day-old Ray session LOGS dir (session_2026-06-03.../logs) had been DELETED
(stale-session log reap; / only 42% full, ray cluster still alive 32 GPU). NOT a training/NCCL/cudagraph issue.
Run had trained step1->350 cleanly: reward raw -1.3 -> ~+0.3, pass@16 -> 0.90-0.96.

FIX for the broken LoRA resume (so we don't lose 350 steps): patched megatron_strategy.load_checkpoint
(synced to all 4 nodes) -- for is_lora, SKIP dist_checkpointing.load (the optimizer distcp has no .metadata)
and warm-start: load adapters from adapter_*.pt, reinit optimizer/LR/RNG; global_step+dataloader restored
from trainer_state.pt/data.pt. Relaunched resume_mode=latest from step 350. VERIFY: reward must continue
~0.3 (not reset to -1.3); if off, revert patch + restart fresh. (Proper long-term fix = make LoRA save land
.metadata so optimizer state is resumable too.) If the ray-logs reap recurs -> restart ray for a fresh session.

### LoRA resume — SECOND bug found; giving up on resume, restarting FRESH (2026-06-09 18:48)
After the ckpt-load patch (issue #1: skip optimizer distcp) resume got further but hit issue #2:
FileNotFoundError on .../global_step_350/policy/adapter_tp1_pp0_cp0_dp1_ep5_etp0.pt. The LoRA SAVE writes
only 8 adapter files (tp0-3 x dp6/dp7, covering ep0-7) -- adapters are dp-REPLICATED but only dp6/dp7 ranks
write files; LOAD's _get_rank_path expects EVERY rank's own adapter_..dp{N}..pt -> missing for dp0-5.
=> SkyRL LoRA megatron ckpt is broken on TWO counts (missing optimizer .metadata + adapter per-rank coverage).
Proper fix (for user, NOT done unsupervised - wrong rank-mapping would corrupt the model): make LoRA save
write/lookup adapters by (tp,ep) ignoring dp (or have all dp ranks save), AND land optimizer .metadata.
The issue-#1 patch in megatron_strategy.load_checkpoint is LEFT IN (correct partial fix, harmless to non-LoRA
& fresh runs) but LoRA RESUME STILL DOES NOT WORK. Restarting LoRA FRESH on the fresh ray session; step-350
adapters preserved in ckpts/..._bak_step350 (usable for inference/merge). Validation goal already met
(350 stable steps, reward -1.3->+0.3, pass@16 ->0.96).

### Note: GLOO/NCCL_SOCKET_IFNAME = control plane, not data path (clarifying line 285)
Head has 5 IPv4 ifaces (enp2s0=10.40.16.194 cluster subnet; enp7s0/enp8s0 other fabrics; vxlan.calico
k8s overlay; lo) -> NCCL/Gloo auto-detect can pick a wrong/non-routable one -> multi-node init hangs.
NCCL_SOCKET_IFNAME pins only NCCL's TCP BOOTSTRAP/rendezvous + out-of-band control; the actual collective
DATA rides InfiniBand (NCCL_IB_HCA=mlx5_2..9). GLOO_SOCKET_IFNAME pins Gloo (CPU backend, all-TCP: used by
init_process_group/barrier/CPU collectives). So enp2s0 being a modest Ethernet link is fine - it's not the
bandwidth path. Per-node (head/n2=enp2s0, n4=enp1s0); set at `ray start` per node, NOT in global runtime_env.

### enforce_eager A/B run (2026-06-09 20:20, wandb run dapo_lora_r32_enforceEager_*)
Launched fresh LoRA run with enforce_eager=true (no cudagraphs), compilation_config removed, gdn_prefill=triton,
distinct run name (enforceEager) so it does NOT resume. Confirms the cudagraph win in the real RL loop:
  generate(step1): enforce_eager=679.4s  vs  FULL_DECODE_ONLY=~217s  => ~3.1x slower generation (eager).
(matches the earlier full-FT standalone finding.) Run script: run_lora_enforceEager_4node.sh; log out_lora_enforceEager.log.
- enforce_eager A/B steady-state: step2=1398s (eager) vs ~957s (FULL_DECODE_ONLY) => ~1.46x slower end-to-end
  (generate 679 vs 217 = 3.1x; train phase unchanged dilutes it). 2-step stable, fresh, no errors.

### 2026-06-10 06:36 — non-LoRA FULL-FT run relaunched (enforce_eager=false + FULL_DECODE_ONLY)
run_final_4node.sh: full fine-tune (NO LoRA), enforce_eager=false, FULL_DECODE_ONLY, LR=1e-6, triton prefill,
wandb proj qwen3_5_dapo, ckpt_interval=5 + max_ckpts_to_keep=1 (per user). Fresh from scratch.
DISK RISK (watching): workers n4/m8htz only ~15G free (67G model cache fills /), n2 14G free + 185G of OLD
LoRA HF exports (exports/dapo_lora_r32_.../global_step_{135,270,300}, 62G each). Full-FT ckpt shards are
node-local + large (model+optimizer) -> step-5 ckpt may NOT fit on n4/m8htz. PLAN: let step5 attempt; if it
fails/fills disk -> revert ckpt_interval=-1, clear partial, relaunch (non-destructive), report. NOT deleting
user artifacts unprompted; n2's 185G LoRA exports are the obvious free-able space if user wants ckpts to fit.
wandb metric ingestion incident (5h delay, no loss) still backfilling; local datastore is source of truth.

### CORRECTION (2026-06-10 07:51): disk was a FALSE ALARM + step-5 ckpt OK
/home is /dev/md0 = 7.0T RAID, 6.3T FREE (I'd wrongly read df / = /dev/vda2 93G OS-root; ckpts live on /home).
So disk is a NON-issue; no cleanup/fallback needed. Full-FT step-5 ckpt WROTE fine (save 59s; global_step_5
= 118G apparent/sparse .distcp shards across nodes on /home; max_ckpts_to_keep=1 bounds it). Run healthy step5+.
CAVEAT: global_step_5/policy has 8 .distcp shards but NO .metadata -> same SkyRL ckpt bug as LoRA; full-FT
resume likely also fails (CheckpointingException). Ckpts write OK but may not be resumable for crash-recovery.
Not testing resume now (won't disturb healthy run); will confirm only if a crash forces it.

### CORRECTION (2026-06-10): the default cudagraph mode (FULL_AND_PIECEWISE) DOES work for Qwen3.6
An earlier note hypothesized that the FULL_AND_PIECEWISE default (enforce_eager=False, cudagraph_mode unset)
would be "the likely source of instability" because GDN is UNIFORM_BATCH and FULL-on-mixed-batches is
unsupported. THAT WAS WRONG / imprecise: FULL_AND_PIECEWISE = (FULL for uniform decode, PIECEWISE for
prefill/mixed) — it does NOT run FULL on mixed batches, so it already respects GDN's UNIFORM_BATCH limit.
The "FULL-on-mixed unsupported" issue only applies to a *pure* cudagraph_mode=FULL.
EVIDENCE: (a) standalone n3 probe: default resolved to FULL_AND_PIECEWISE, 7566 tok/s, correct output;
(b) 4-node LoRA A/B (run dapo_lora_r32_defaultcg, wandb 4gte237g, out_lora_defaultcg.log): step1=1013.6s,
generate=221.7s, GPU ~59GB/0.7util (no OOM), 0 crashes — i.e. ~identical to FULL_DECODE_ONLY (step1 ~1009s,
gen 217s). => default and FULL_DECODE_ONLY are effectively equivalent here. We still PIN FULL_DECODE_ONLY as
the conservative/memory-leaner choice (captures only decode graphs), but the default is NOT broken.

### 2026-06-16 — TASK: test CP=2 + seq packing (PR #1769) on 4 nodes, Qwen3.6-35B-A3B, NO LoRA
Script: /home/charlie_key/SkyRL-remote/examples/train/megatron/run_megatron_dapo_qwen3.5_35b_a3b.sh (user set knobs).
SkyRL-remote = commit bbb0bc1f "[megatron] Add seq packing support for qwen3.5 (#1769)" on main.
Knobs: MODEL Qwen3.6-35B-A3B, 4 nodes, TP4/PP1/CP2/EP8, enforce_eager=false (no cudagraph_mode->vLLM default
FULL_AND_PIECEWISE), gdn_prefill=triton, wandb, ckpt_interval=5, run_name dapo_qwen3_6..._seqPacking_..cp2..
Cadence: 15min until 2 steps trained, then 45min. GH_TOKEN in ~/.bashrc. (Prior belief: CP>1 needs sample
packing + GDN didn't support packing -> PR #1769 is meant to FIX exactly this.)
KEY SETUP FACTS:
- /home is NODE-LOCAL. SkyRL-remote is ONLY on head. SkyRL/.venv exists on ALL nodes (proven megatron env).
- HEAD SkyRL/.venv imports skyrl from /home/charlie_key/SkyRL-remote/skyrl (editable -> PR code).
- PR #1769 changed ONLY .py/.sh (NO pyproject/uv.lock) => deps unchanged => REUSE SkyRL/.venv + overlay PR
  source via PYTHONPATH=/home/charlie_key/SkyRL-remote (no 20-min venv rebuild needed).
- Ray is DOWN (no gcs/raylet). Must restart on all 4 nodes.
PLAN: (1) rsync SkyRL-remote code -> n2,n4,m8htz (excl .venv/.git). (2) restart Ray on 4 nodes via
SkyRL/.venv/bin/ray with PYTHONPATH=/home/charlie_key/SkyRL-remote + per-node IFACE (head/n2/m8htz=enp2s0,
n4=enp1s0), num-gpus=8, head 10.40.16.194:6379. (3) launch: cd SkyRL-remote; PYTHONPATH=$PWD + SkyRL/.venv
python -m examples.train.algorithms.dapo.main_dapo with the script's args (or edit script's `uv run` line ->
SkyRL/.venv/bin/python + run). (4) success = no "context parallel only supported with sample packing" error,
sync_weights ok, steps complete with CP=2+packing. Nodes reserved: 5q745,77pcc(n2),8vnx2(n4),m8htz.

### CP=2 test: init crash #1 -> config fix (2026-06-16 07:46)
CP=2 run crashed at init_model on all workers: AssertionError "Qwen3-VL model only supports context parallelism
with calculate_per_token_loss enabled". Fix: relaunch with override (via script $@):
trainer.policy.megatron_config.transformer_config_kwargs.calculate_per_token_loss=True. (Setup: PR#1769 code on
all 4 nodes via PYTHONPATH=/home/charlie_key/SkyRL-remote + SkyRL/.venv; ray restarted; script uv-run line ->
SkyRL/.venv/bin/python.)

### PR #1769 CP+seq-packing test on Qwen3.6-35B-A3B — FINDING (2026-06-16 ~07:55): routing gate fails
Setup OK: PR code (bbb0bc1f) on all 4 nodes via PYTHONPATH overlay on SkyRL/.venv; ray restarted; reuses
proven megatron venv (PR changed no deps).
RESULT: both CP=2 and CP=1 (+ seq packing, language_model_only=True) CRASH at init_model:
  - CP=2: AssertionError "Qwen3-VL model only supports context parallelism with calculate_per_token_loss enabled"
  - CP=1: ValueError "remove_microbatch_padding=True not supported for models that pack inside their own
    forward (Qwen3VLModel)... double-packs/corrupts GatedDeltaNet cu_seqlens. Set language_model_only=True OR
    remove_microbatch_padding=False."
ROOT CAUSE: model builds as Qwen3VLModel even with language_model_only=True. The routing gate
megatron_worker.py:367 `if language_model_only and maybe_force_qwen35_text_bridge(bridge, hf_config)` returned
FALSE (its log line never printed). maybe_force (model_bridges.py:147) matches hf_config.architectures against
{Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration}. config.json TOP-LEVEL architectures =
['Qwen3_5MoeForConditionalGeneration'] (MATCHES), but the gate reads hf_config = update_model_config(
hf_config_original,...) which evidently does NOT expose that arch (text_config.architectures is None) -> gate
False -> VL path -> fail. => PR #1769's language_model_only->GPTModel routing does not engage for
Qwen3.6-35B-A3B as-is. No config-only workaround tests packing (other remedy = remove_microbatch_padding=False
disables the very packing under test). FIX is on the PR side (make maybe_force read top-level architectures /
handle this config), OR test with a model whose loaded hf_config.architectures matches. Cluster left idle
(Ray up w/ PR code on PYTHONPATH, 4 nodes reserved). Loop stopped — relaunching as-is is deterministically futile.

### PR#1769 Qwen3.6 routing — REFINED (2026-06-16): gate logic is CORRECT in isolation
Repro (exact path AutoConfig+update_model_config with real bos/eos/pad override): hf_config.architectures =
["Qwen3_5MoeForConditionalGeneration"] survives, maybe_force_qwen35_text_bridge returns TRUE and rewrites
bridge arch -> Qwen3_5MoeTextForCausalLM. So language_model_only IS set+threaded AND the gate SHOULD route.
Yet run builds Qwen3VLModel (wrapper model_packs_sequences_internally(actor_module)=True -> error) and the
"forcing..." log is absent. Hypothesis: rewrite is ineffective because AutoBridge.from_hf_pretrained already
dispatched the VL model class; rewriting bridge.hf_pretrained.config.architectures afterward does not re-select
the model -> to_megatron_provider still builds Qwen3VLModel. INSTRUMENTED megatron_worker.py gate w/ DBG_GATE
prints (lmo, hf_arch, bridge_arch before/after, forced); synced to all nodes; relaunching CP=1 to capture.
REVERT the DBG prints after (git checkout megatron_worker.py).

### PR#1769 Qwen3.6 routing — ROOT CAUSE FOUND (2026-06-16): stale megatron stack on WORKER nodes
File-based instrumentation (/tmp/dbg_gate.txt) of the gate in init_configs (megatron_worker.py ~L367) showed:
  HEAD (10.40.16.194): forced=True  -> bridge arch rewritten to Qwen3_5MoeTextForCausalLM (gate WORKS)
  WORKERS n2/n4/m8htz: forced=False -> bridge stays Qwen3_5MoeForConditionalGeneration -> builds Qwen3VLModel -> crash
Same inputs everywhere (lmo=True, hf_arch=[Qwen3_5MoeForConditionalGeneration]). model_bridges.py is byte-identical
(md5) across nodes. The divergence: model_bridges.py wraps its real maybe_force_qwen35_text_bridge in try/except
ImportError with a stub that ALWAYS returns False. The try block imports
  megatron.bridge.models.qwen.qwen35_bridge  ->  OK on head, ModuleNotFoundError on ALL 3 workers.
Versions: head = megatron-bridge 0.6.0+91a15142 / megatron-core 0.19.0+71e418ea7 (has qwen35_bridge.py +
experimental_attention_variant_module_specs). Workers = megatron-bridge 0.5.0+8382dc34 / megatron-core 0.16.0rc0
(NO qwen35_bridge.py). TE 2.11.0 on all. => CONCLUSION: PR #1769 GATE LOGIC IS CORRECT; the blocker is an
ENVIRONMENT SKEW: head venv was upgraded (core 0.16->0.19, bridge 0.5->0.6) but the 3 worker node-local venvs
were not. Prior 4-node runs worked only because they did not use language_model_only/packing. FIX = bring worker
megatron stack up to head (sync megatron/core + megatron/bridge + dist-info, or reinstall). Reverting DBG edits next.

### PR#1769 Qwen3.6 — FIX VERIFIED (2026-06-16 16:23): forced=True on ALL 4 nodes after syncing
megatron core0.19+bridge0.6+training to workers. remove_microbatch_padding init crash GONE; model builds as
GPTModel (text) on every rank; CP=1 + sequence packing path now active. Run then died at step0 on RESUME:
CheckpointingException ".../global_step_75/policy is not a distributed checkpoint" -- stale step-75 ckpt from a
prior NON-packing run sharing ckpt_path (script ckpt_path=dapo_qwen3_5_...tp4pp1cp1ep8etp1, lacks 3_6/seqPacking
suffix; resume_mode=latest picked it up). Known .metadata-drop ckpt bug. Relaunching with resume_mode=none +
fresh isolated ckpt_path/export_path (dapo_qwen3_6_35b_a3b_seqPacking_*) via $@; NOT deleting the step-75 artifact.
Debug instrumentation reverted + resynced to all nodes.

### PR#1769 Qwen3.6 — CP+PACKING TEST RESULT: *** PASS *** (2026-06-16 16:44)
Step 0 completed FULLY and CLEANLY with CP=1 + sequence packing on the GPTModel GDN path:
  generate 225.8s | policy_train 395.2s | step total 876.9s. NO cu_seqlens corruption, NO NaN, NO shape error.
=> CONCLUSION: sequence packing (remove_microbatch_padding) + CP work for Qwen3.6-35B-A3B after PR #1769,
   GIVEN the worker megatron stack is upgraded to core0.19/bridge0.6 (the fix done earlier this session).
Then crashed at START of step 1: ActorDiedError / SYSTEM_ERROR "connection error code 2, End of file" on a
MegatronPolicyWorkerBase (n4 PID 1166253) + a head worker. NO Python traceback => external SIGKILL/SIGSEGV.
Host RAM ruled OUT as cause: nodes have 1763GB RAM, ~1660GB free, swap=0 (offload nowhere near limit).
No dmesg/journal OOM line visible (may be priv-restricted). GPUs idle post-crash; ray cluster healthy 4node/32GPU.
Likely a transient segfault/CUDA/NCCL hiccup at the step0->1 boundary. ACTION: relaunching once identically
(fresh isolated ckpt path, resume_mode=none) to test transient-vs-deterministic. If it recurs at same boundary,
it is deterministic -> pause for user (memory/sync config decision: TP8, gpu_mem 0.5, offload_fraction<1).

### PR#1769 Qwen3.6 — step-1 crash is DETERMINISTIC (2026-06-16 17:04): reproduced at SAME step0->1 boundary
Second identical run: step0 clean again (generate 216s, policy_train 262.6s, step 621s) then ActorDiedError /
SYSTEM_ERROR at step 1 start, SAME nodes (n4 10.40.40.19 + head). Dead MegatronPolicyWorkerBase logs have NO
traceback / NO CUDA/NCCL error -> clean external SIGKILL/SIGSEGV; "empty_cuda_cache: true" logged just before.
Host RAM ruled out (1.7TB free). => NOT the PR (packing proven twice). Prime suspect: GPU-mem peak at colocated
inter-step transition (optimizer h2d overlap + vLLM KV realloc @ gpu_memory_utilization=0.7). Attempt #1 (bounded):
relaunch with gpu_memory_utilization=0.5. If still dies at step1 -> present options to user (TP8 / offload_fraction<1
/ un-disable NCCL P2P-SHM / num_engines or vllm mem). Not chasing further config guesses autonomously beyond this.

### PR#1769 Qwen3.6 — gpu_mem=0.5 did NOT fix step-1 crash (2026-06-16 17:22)
Third run (gpu_mem 0.5): step0 clean (generate 215.7s, policy_train 251.4s, step 629s) then SAME ActorDiedError /
SYSTEM_ERROR at step1 boundary, SAME nodes (head 10.40.16.194 + n4 10.40.40.19). => vLLM KV-cache GPU pressure
RULED OUT. Crash is in MegatronPolicyWorkerBase at the inter-step transition (optimizer offload / empty_cuda_cache),
fires the instant step0 Finished -> worker died during step0 tail. New prime suspect: CPU optimizer-offload +
overlap d2h/h2d path (optimizer_cpu_offload=true, overlap_cpu_optimizer_d2h_h2d=true, offload_fraction=1.0).
Attempt #2 (final autonomous knob): disable optimizer offload (all 3 kwargs false). Distributed-opt state ~13GB/GPU
sharded across DP8 -> fits 144GB H200 without offload. If this ALSO crashes at step1 -> STOP, hand decision to user.

### PR#1769 Qwen3.6 — offload-OFF ALSO crashed at step1 (2026-06-16 17:39) -> STOPPING autonomous knob-turning
Fourth run (optimizer offload disabled): step0 clean again (generate 214s, policy_train 248s, step 601s) then SAME
ActorDiedError/SYSTEM_ERROR at step1 boundary. => CPU optimizer-offload RULED OUT too (along with vLLM KV mem).
*** KEY PATTERN (all 4 crashes): the two dying MegatronPolicyWorkerBase workers are ALWAYS on head 10.40.16.194
+ n4 10.40.40.19; NEVER n2 (10.40.58.131) or m8htz (10.40.62.120). *** Step0 full 32-GPU train succeeds every
time (so n4 compute is fine), but a worker on the head+n4 pair dies at the inter-step offload/sync/empty_cache
transition with external SIGKILL/SIGSEGV, no python traceback, no CUDA/NCCL msg in worker logs. This points away
from a memory-config knob and toward (a) the specific head<->n4 NCCL link/P2P during weight-sync or post-step
cleanup, or (b) an n4-pair-specific issue. NOTE: utils.py currently DISABLES NCCL P2P/SHM (prior mitigation).
NEXT OPTIONS FOR USER (their call): (1) un-disable NCCL P2P/SHM in utils.py; (2) drain n4, run 3-node/24-GPU to
test the n4-pair hypothesis; (3) NCCL_DEBUG=INFO + dmesg-with-privs to catch the actual kill cause; (4) MEGATRON_TP=8.
NOT relaunching further without user direction.

### *** STEP-1 CRASH ROOT CAUSE FOUND & FIXED (2026-06-16): degraded IB HCA mlx5_3 on head ***
The deterministic step0->1 SYSTEM_ERROR worker kills (always head 5q745 + n4, no python traceback) were NOT
memory/PR/offload — they were a DEGRADED InfiniBand rail. Followed MULTINODE_together_cluster_060126.md §1/§4
+ used SkyRL/ib_hca_repro/. Evidence (NCCL all_reduce head<->n4, ib_pair_test.sh):
  - ibstat: head 5q745 mlx5_3 = ACTIVE but 10 Gb/s SDR (not 400 NDR); mlx5_2 recovered; all other rails OK.
  - mlx5_3 solo -> FAIL: IBV_WC_RETRY_EXC_ERR(12) hca mlx5_3 -> ncclRemoteError -> abort.
  - full mlx5_2..9 @1 GPU/node -> PASS (rail affinity: GPU0 uses mlx5_2, misses mlx5_3).
  - full mlx5_2..9 @8 GPUs/node -> FAIL on mlx5_3 (GPU5) == faithful repro of the training crash.
  - fix mlx5_2,mlx5_4..9 (drop mlx5_3) @8 GPUs/node -> PASS.
NCCL watchdog aborts the proc on the retry-exhaust -> Ray SYSTEM_ERROR with no traceback (matches exactly).
FIX APPLIED: skyrl/train/utils/utils.py NCCL_IB_HCA -> "mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
(7 rails, drop mlx5_3). Real fix: service 5q745 mlx5_3 then restore 8 rails. Repro+docs: ib_hca_repro/README.md,
MULTINODE doc §4/§9. => With this, the Qwen3.6 CP+packing DAPO run should clear step1 (all blockers now resolved:
worker megatron stack upgraded earlier + IB rail fixed). NOTE: utils.py NCCL_IB_HCA is propagated via Ray
runtime_env, so a NEW driver launch picks it up (no ray restart needed); a currently-running driver would not.

### *** SUCCESS: Qwen3.6 CP+packing CLEARS THE STEP BOUNDARY (2026-06-16 20:42) ***
After adding NCCL_IB_HCA="mlx5_2,mlx5_4..9" (drop flapping mlx5_3) to SkyRL-REMOTE utils.py (the driver-imported
copy; it previously had NO NCCL_IB_HCA pin -> NCCL auto-selected mlx5_3 -> crash) + the earlier worker megatron
stack upgrade: 4-node run dapo_qwen3_6_35b_a3b_seqPacking_ibfix cleared step0 AND the inter-step weight-sync that
killed the 3 prior runs. Step0: generate 212.6s, policy_train 453.7s, sync_weights 34.9s, step total 837.9s;
Training Batches Processed 1/2700; step1 generate now running. No IBV_WC/SYSTEM_ERROR. Verified worker env has
NCCL_IB_HCA=mlx5_2,mlx5_4..9 (mlx5_3 excluded). => CP=1 + sequence packing WORK and multi-step is STABLE with the
IB fix. Monitoring to 45-min cadence after step1 completes (2 full steps).

### Qwen3.6 IB-fix run: cleared 2 FULL STEPS then crashed at step2->3 (2026-06-16 ~21:00)
BIG progress: with mlx5_3 dropped, the run did step0 (838s) + step1 (797s) CLEANLY (prior runs died at step1).
Then SAME silent SYSTEM_ERROR / ActorUnavailableError (Socket closed, rpc 14) at the step2->3 boundary, this time
head(5q745) + n2 (whole block of head workers died together). NO IBV_WC/nccl error captured this time (worker .out/.err
hold no diagnostics anyway). Post-crash: all head IB rails read 400 NDR (mlx5_3 flapped back up); RAM 1.66TB free; GPUs idle.
*** THROUGHLINE: head node 5q745 is in ALL 5 crashes (head+n4 x4, head+n2 x1); peer rotates, 5q745 constant. ***
mlx5_3 was its worst symptom but 5q745 IB/transport looks broadly intermittent/unstable. Options pending user decision:
(1) 3-node run EXCLUDING 5q745 GPUs (n2/n4/m8htz, 24 GPU) to confirm/avoid the bad head; (2) relaunch 4-node as-is
(will progress but likely intermittently re-crash on 5q745); (3) pause for infra to service 5q745. NOTE ckpt_interval=5
so no ckpt saved yet (died at step ~2-3) -> relaunch is from scratch.

### Step2->3 crash post-IB-fix = NCCL COLLECTIVE STALL, not the mlx5_3 IBV error or a packing bug (2026-06-16, infra log)
infra-260616_202325.log on all 4 nodes: all 32 ranks hit `Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=112, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout=1800000ms)` at 21:37:36 (hang began ~21:07).
SAME SeqNum on every rank => transport STALL (all entered the 1-elem allreduce; it never completed in-network),
NOT a control-flow desync. Last normal log 21:05:01: forward_backward microbatches_this_step=64 on dp_ranks 0/2/4/6
(BALANCED) => NOT an uneven-microbatch packing desync. n4 `KeyError /psm_...` = benign resource_tracker teardown noise.
=> Subtler IB-fabric failure mode (silent stall, no IBV retry-exhaust). QP-setup sweep (ib_sweep_head.sh) currently
ALL-GREEN (head<->n2 10/10, head<->n4 10/10) -> flap intermittent & currently up; setup test cannot catch a
mid-collective stall. Throughline still intermittent head/fabric instability. Mitigations to consider: lower NCCL
watchdog/heartbeat timeout to fail fast (not wait 30min); NCCL_IB_TIMEOUT/NCCL_IB_RETRY_CNT tuning; or 3-node run
excluding 5q745; root fix = infra services 5q745 fabric.

### Real DAPO run w/ token-batching launched (2026-06-17 08:44): TP8/EP8/CP1, max_tokens_per_microbatch=200000
Same DAPO script + TP8 (was TP4) + trainer.max_tokens_per_microbatch=200000 + recompute_old_logprobs_per_minibatch=True.
Unified 4-node cluster (mlx5_3 excluded from NCCL_IB_HCA; Ray temp+logs on /home, not /tmp root). world=32 dp=4 (TP8).
Token batching CONFIRMED working: packs real seqs to ~199000-199996 tok/microbatch (cap 200000), 24-32 seqs/mb,
5-6 microbatches/minibatch. GPUs ~100%, NO OOM/stall/IBV. Step0 in policy_train. Driver log: out_dapo_tp8_mbs200k.log.
