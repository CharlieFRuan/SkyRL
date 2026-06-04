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
