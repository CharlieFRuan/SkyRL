# max_tokens_per_microbatch tuning — Qwen3.6-35B-A3B (Megatron, 1 node / 8×H200)

Goal: find the **maximum `max_tokens_per_microbatch`** we can run without OOM under the Megatron
config (token-based micro-batching, PR #1477 — already merged in SkyRL-remote `eda11272`), on top of
`remove_microbatch_padding=True`, with `recompute_old_logprobs_per_minibatch=True`. Target ceiling
**262144 tokens** (= 1024 seqs × 256). Then, if 262144 OOMs, tune Megatron knobs (1 node only) to fit it.

Harness: `examples/train_scripts/full_context/run_full_ctx_qwen36.sh` (dummy `FullCtxTrainer` —
fabricates fixed-length sequences, runs `num_dummy_steps=3`, no real generation). All code is
SkyRL-remote (run with `PYTHONPATH=/home/charlie_key/SkyRL-remote`; deps from `SkyRL/.venv`).

## Key facts
- **`max_tokens_per_microbatch` is PER DP RANK** (applied to each data-parallel replica's shard), not
  per physical GPU. Within a rank the TP (and CP) group jointly processes the microbatch: with TP8 the
  262144 tokens' activations are sharded across the 8 TP GPUs; with CP>1 the *sequence* is additionally
  split across CP GPUs.
- **`micro_train/forward_batch_size_per_gpu` are IGNORED at runtime when `max_tokens_per_microbatch>0`**
  (token bin-packing forms microbatches — `megatron_worker.py:985`). But `validate_batch_sizes` still
  asserts `mini_per_gpu % micro == 0` at startup, so they must stay valid divisors → kept at `1`.
- Batch shape: `train_batch=mini_batch=128` prompts, `n_samples=16`, seq=256 (`max_prompt=128`+`max_gen=128`).
  Total = 128·16·256 = **524288 tokens**. With DP=1 that's 524288 tok/rank → packs into 2 microbatches
  of 262144 at `max_tokens=262144`. (n_samples=16 = headroom "just in case"; n_samples=8 would give
  exactly 262144/rank when DP=1.)
- Defaults already match the 128k sweep: `recompute_granularity="full"`, `recompute_modules=["core_attn"]`,
  `moe_token_dispatcher_type="alltoall"`, `moe_grouped_gemm=True`, optimizer offload fraction 1.0.
- **Effective microbatch size** is logged by `worker_utils.py TokenBasedBatchIterator`:
  `[microbatch-pack] rank=… max_tokens_per_microbatch=… num_microbatches=… seqs_per_microbatch=[…]
  tokens_per_microbatch=[…] peak_tokens=…`.

## Parallelism constraints (1 node, world=8)
- `world(8) = TP × PP × CP × DP`  → `DP = 8/(TP·PP·CP)`.
- MoE: `world(8) % (ETP × EP × PP) == 0`. With ETP1,PP1: `EP ∈ {1,2,4,8}`; `EP·DP_moe = TP·CP·DP`.
  EP8 = most expert-memory-efficient (each GPU holds 1/8 of experts); prefer EP8 for max context.
- **CP shards the sequence dim** → the most direct lever for fitting long microbatches; CP trades
  against TP for the 8-GPU budget (`TP4×CP2=8`, `TP2×CP4=8`). **CP>1 for this GDN model REQUIRES
  `trainer.policy.megatron_config.transformer_config_kwargs.calculate_per_token_loss=True`.**

## Phase 1 — TP8/EP8/CP1 (128k-sweep winner), binary search
Known from 128k sweep: TP8/EP8 OK at 131072 (step2 100.8s); TP4/EP8 OOMed at 128k. So search [131072, 262144].
```
bash run_full_ctx_qwen36.sh                                            # max_tokens=262144 (default)
bash run_full_ctx_qwen36.sh trainer.max_tokens_per_microbatch=196608  # if 262144 OOMs (midpoint)
...
```

| max_tokens | TP/EP/CP | result | step time | peak_tokens (log) | notes |
|-----------:|----------|--------|-----------|-------------------|-------|
| 131072 | 8/8/1 | OK (prior 128k sweep) | 100.8s | — | WINNER_128K baseline |
| 131072 | 8/8/1 | **OK** (this harness) | ~75s/step | 131072 (×4 mb) | floor confirmed |
| 196608 | 8/8/1 | **OK** | ~97s/step | 196608 (×3 mb) | TP8 max in (196608, 262144) |
| 229376 | 8/8/1 | **OK** | ~95s/step | 229376 (×3 mb) | TP8 max in (229376, 262144) |
| 245760 | 8/8/1 | **OK** | ~110s/step | 245760 (×3 mb) | TP8 max in (245760, 262144) |
| 253952 | 8/8/1 | **OOM (marginal, step2)** | — | — | frag; expandable_segments ALREADY on by default |
| 262144 | 8/8/1 + env expandable | **OOM (step1)** | — | — | hard capacity limit, NOT fragmentation |
| 262144 | 8/8/1 | **OOM** | — | (Triton CUDA OOM in forward_backward, step1) | TP8 cannot fit 262144 |

## Phase 2 — if 262144 OOMs at TP8: 1-node configs to extend max context
All DP=1, EP8 (expert-mem-optimal), `calculate_per_token_loss=True` for CP>1. Fix `max_tokens=262144`.
Append e.g.:
```
bash run_full_ctx_qwen36.sh \
  trainer.policy.megatron_config.tensor_model_parallel_size=4 \
  trainer.policy.megatron_config.context_parallel_size=2 \
  trainer.policy.megatron_config.transformer_config_kwargs.calculate_per_token_loss=True
```

| config (TP/EP/CP) | seq shard | trade-off | result | step time |
|-------------------|-----------|-----------|--------|-----------|
| 8/8/1 | ×1 | most TP weight-sharding, no seq shard | **OOM @262144** | — |
| 4/8/2 | ×2 | CP halves activation; TP4 less weight-shard | _pending_ | |
| 2/8/4 | ×4 | CP quarters activation; TP2 more weight/GPU | _pending_ | |
| 4/4/2 | ×2 | (user suggestion) EP4 = experts replicated 2× (more expert mem) | _pending_ | |

Most promising for max context = highest CP (TP2/EP8/CP4) since CP directly shards the long-sequence
activations. If still OOM at 262144 with all 1-node configs, the remaining lever is more aggressive
recompute (`recompute_modules` += mlp/moe) at a compute cost.

## Results / takeaways
- **TP8/EP8/CP1 max ≈ 245760** (reliable; 253952 OOMs at step2 by ~0.25GiB = fragmentation; 262144 OOMs at step1).
- **expandable_segments is ALREADY ON by default** (`trainer.use_expandable_segments=True`, config.py:669; applied post-init via torch allocator settings — activations expandable, weights kept IPC-safe). Forcing it via env (PYTORCH_CUDA_ALLOC_CONF) is redundant for activations and unsafe for colocate IPC. 262144 OOMs at *step1* even with full env expandable -> hard capacity limit on TP8, not fragmentation.
- **PHASE-1 CONCLUSION: TP8/EP8/CP1 reliable max ≈ 245760 tokens/microbatch** (196608, 229376, 245760 OK; 253952 marginal step2 OOM; 262144 step1 OOM). 262144 NOT reachable on TP8/EP8/CP1.

- **PyTorch OOM `expandable_segments` hint is UNCONDITIONAL on torch 2.11.0+cu128** (verified: present even with env `expandable_segments:True` and with runtime toggle). So the hint in the 253952 OOM does NOT mean it was off — `trainer.use_expandable_segments=True` was active by default. The 253952/262144 OOMs are genuine capacity, not a missing expandable flag.
- Logging: each probe now -> unique /home/charlie_key/out_maxtok_<cfg>.log (earlier runs shared one path and overwrote).
- NEXT: Phase 2 — reach 262144 via CP (shard sequence).
- **CP REQUIRES TWO KNOBS** (else AssertionError 'Cannot average in collective when calculating per-token loss' at init):
  `transformer_config_kwargs.calculate_per_token_loss=True` AND `ddp_config.average_in_collective=False` (default True, config.py:157).
- **Phase-2 parallelized across the 4 nodes** (each config = 1 node/8 GPU; 4 jobs fill 32 GPU). Round 1 candidates @262144:
  TP4/CP2/EP8, TP2/CP4/EP8, TP4/CP2/EP4, TP1/CP8/EP8. Logs: out_maxtok_<cfg>_262144.log.

### Parallel attempt #1 FAILED (contention) 2026-06-17 ~08:04
Launched 4 CP configs concurrently on the ONE shared 4-node Ray cluster -> all 4 OOMed at step1 (each tried
30.31GiB with only 8-29GiB free). Workers scattered across overlapping nodes (all touched head .194) -> jobs
shared GPUs -> contention OOM, NOT real config limits. FIX: run 4 INDEPENDENT single-node Ray clusters
(one ray --head per node, 8 GPU each), one job per node. Re-running candidates that way.

### Parallel attempt #2 — 4 INDEPENDENT 1-node Ray heads (one per node), no contention 2026-06-17 ~08:20
Fix vs attempt#1: each node runs its own `ray --head` (8 GPU); launch one job per node via ssh -> launch_maxtok.sh.
Gotcha: dapo parquet only existed on head -> 3 ssh jobs hit FileNotFoundError at dataloader build. Fixed by rsync
of ~/data/dapo to n2/n4/m8htz, relaunched.
RESULTS @262144 (isolated, real):
- TP4/CP2/EP8 -> **OOM** (tried 30.31GiB, 28.78 free; ~1.5GiB short — real, marginal). CP2 activation savings
  offset by TP4's larger weight share vs TP8.
- TP2/CP4/EP8, TP4/CP2/EP4, TP1/CP8/EP8 -> rerunning (most hopeful = TP1/CP8/EP8, seq ÷8).

### *** PHASE-2 CONCLUSION (2026-06-17 ~08:32): 262144 NOT reachable on 1 node by ANY TP/CP/EP ***
All 4 isolated configs OOM at step1 trying the SAME 30.31 GiB (invariant to TP/CP/EP):
  TP4/CP2/EP8 (28.78 free), TP2/CP4/EP8 (23.52), TP4/CP2/EP4 (8.07), TP1/CP8/EP8 (15.36) — none >=30.31 free.
ROOT CAUSE: backward of the logprob op `from_parallel_logits_to_logprobs` (model_utils.py:262,
`grad_input = torch.cat(all_grad_input, dim=1)`) materializes the full [tokens, vocab/TP] grad. Its size =
tokens * vocab / (TP*CP). On 1 node TP*PP*CP*DP=8 and DP1/PP1 => TP*CP=8 FIXED => this tensor is the SAME ~30.31GiB
for every split. CP trades seq-shard for smaller TP vocab-shard; product TP*CP=8 unchanged => no help.
=> 1-node max_tokens_per_microbatch ~= 245760 (TP8/CP1, phase 1). To go higher:
   (a) LOWER max_tokens (245760 fits); or
   (b) CODE FIX: chunk/accumulate the logprob backward so it never materializes the full torch.cat at L262
       (it already chunks over seq for the forward path but re-concatenates the full grad) -> would cut this peak; or
   (c) MULTI-NODE: TP*CP>8 shrinks the logprob-grad proportionally (e.g. 2 nodes TP8/CP2 -> half).
Note: TP8/CP1 is the BEST 1-node config (most vocab+weight sharding => most free mem). Lower TP is strictly worse here.

### TP8 DAPO run takeaway + recompute note (2026-06-17 ~09:30)
TP8/EP8 mbs=200000 real DAPO: world=32 dp=4. Steady-state (step1) ~697s/step (generate 218 + policy_train 312 +
overhead); step0 1046s was warmup. vs mbs2 (TP4/EP8 dp=8) ~600s/step, policy_train <300s. So TP8 is ~15% slower/step.
DP halved 8->4 (TP4->TP8); but compute ~balances (DP4*TP8 ~ DP8*TP4 per-GPU work) so slowdown is mostly TP8 comm
(8-way TP all-reduce) + token-batching, NOT DP. recompute_old_logprobs_per_minibatch only affects the FORWARD_LOGPROBS
phase (runs old-logprob fwd per-minibatch instead of one full-batch fwd -> modestly slower there); does NOT change
policy_train. Keeping it True per user.
TRADEOFF (4 nodes): max_tokens ceiling ∝ tokens/(TP*CP). TP4 fits ~half of TP8. DP8 (fast TP4) and a 200k microbatch
are mutually exclusive. User chose: go back to TP4/EP8 (faster, DP8) with a SMALLER max_tokens from a fresh sweep.

### Phase-1 redo for TP4/EP8 (no CP) — PARALLEL sweep across 4 independent 1-node Ray clusters (2026-06-17 ~09:35)
Killed TP8 run, tore down unified cluster, brought up 4 independent single-node heads (TMPDIR on /home).
1-node TP4/EP8/CP1 -> DP2 (conservative vs the real 4-node DP8 which has more optimizer headroom -> sweep max is a
LOWER BOUND for the real run; safe to use with headroom). n_samples=16 -> 262144 tok/rank available.
Parallel probes @ max_tokens: 81920 (5q745), 98304 (n2), 114688 (n4), 131072 (m8htz). Logs node-local
out_maxtok_tp4ep8_*.log. Goal: find TP4/EP8 max -> pick ~80% for the DAPO run -> re-form unified cluster -> launch
TP4/EP8 DAPO. NO CP.

### TP4/EP8/CP1 sweep RESULTS (1-node DP2)
Round 1: 81920 OK, 98304 OK, 114688 OK, 131072 OOM -> max in (114688, 131072).
Round 2 (refine, parallel): 118784, 122880, 126976, 129024 -> running.
Plan: TP4/EP8 DAPO run max_tokens ~= 80% of the found max (analogous to TP8's 200000/245760). Likely ~100000.
(1-node DP2 is conservative vs the real 4-node DP8 which shards optimizer 8x -> more headroom -> safe.)

### *** TP4/EP8/CP1 MAX FOUND (2026-06-17 ~10:18): ~122880 (1-node DP2) ***
Round 2: 118784 OK, 122880 OK, 126976 OOM, 129024 OOM -> max in (122880, 126976).
(TP8 was ~245760; TP4 ~half, as predicted: logprob-grad ∝ tokens/(TP*CP), TP4/CP1 = 2x TP8/CP1 per token.)
=> DAPO run choice: TP4/EP8, max_tokens_per_microbatch=100000 (~81% of 122880, mirrors TP8's 200000/245760).
Real run is 4-node DP8 (optimizer sharded 8x vs DP2's 2x) -> MORE headroom than this sweep -> 100000 is safe.
Now: tear down 4 heads -> re-form unified 4-node cluster -> edit DAPO script TP8->TP4, mbs 200000->100000 -> launch.

### TP4/EP8 DAPO run LAUNCHED (2026-06-17 ~10:20): max_tokens=100000
Unified 4-node cluster re-formed (mlx5_3 excluded, /home temp). Script edited TP8->TP4, mbs 200000->100000.
world should be 32, dp=8 (TP4). Expect FASTER than TP8 (DP8 + 4-way TP comm). Driver log: out_dapo_tp4_mbs100k.log.
Monitoring 10-min until 2 steps, then hourly.

### TP4 DAPO first launch hit stale-ckpt collision (2026-06-17 10:24) -> relaunched
resume_mode=latest + ckpt_path .../tp4_pp1_cp1_ep8_etp1 had an old broken global_step_75 (CheckpointingException,
the known .metadata-drop bug). Relaunched with resume_mode=none + fresh ckpt/export path
dapo_qwen3_6_35b_a3b_tp4ep8_mbs100k. (Same fix used for the seqPacking runs.)

### *** TP4/EP8 mbs=100000 DAPO STABLE — 2 steps, throughput comparison (2026-06-17 ~11:00) ***
world=32 dp=8. Step0 (warmup): policy_train 403s, step 832s. Step1 (steady): policy_train 286.66s, step 657.13s.
COMPARISON (steady-state policy_train / whole step):
  mbs2 TP4/dp8 micro=2:   <300s / ~576-619s
  TP8/dp4 mbs=200000:      312s  / 697s
  TP4/dp8 mbs=100000:      287s  / 657s   <- THIS
=> Switching back to TP4 recovered policy_train speed (287s, faster than TP8 312s, ~= mbs2). Token-batching at TP4
does NOT slow policy_train. Whole-step 657s (between mbs2 600 and TP8 697); residual vs mbs2 = recompute_old_logprobs
+ token-batching overhead in forward_logprobs (kept True per user). No OOM/errors. Run stable -> hourly monitoring.

### TP4 DAPO crashed at step 155 (2026-06-18 01:49) — EXTERNAL nccl-tests interference; HOLDING relaunch
Ran 155 clean steps, then RuntimeError VLLMInferenceEngine.wake_up() CUDA OOM (cumem_allocator.cpp:139) on n4.
Cause: INFRA is running multi-node NCCL fabric benchmarks (mpirun all_reduce_perf -b 8G -e 16G -g 8, cycling on
head/n2/...) — grabs all 8 GPUs/node, starving the colocated vLLM wake. Not a config fault (155 steps were stable).
Checkpoints (step 145/150/155) are NOT resumable: policy/ dist-ckpt missing .metadata (known io.local_work_dir
temp-move bug) -> resume would throw CheckpointingException. So a relaunch = fresh (lose 155 steps).
DECISION: do NOT relaunch while infra benchmarks run (would collide + pollute their fabric tests). Poll ~20min;
relaunch fresh (TP4/EP8 mbs=100000, resume_mode=none) once nccl-tests/mpirun are gone. Ray cluster still up (4 nodes).
FLAG for user: (1) infra active on cluster; (2) checkpoints non-resumable (.metadata bug) — real blocker for long runs.

### Relaunched after infra benchmarks cleared (2026-06-18 02:15)
Infra nccl-tests/mpirun gone, GPUs free. Relaunched TP4/EP8 mbs=100000 fresh (resume_mode=none, ckpt path
..._r2 since step-155 ckpt non-resumable). Auto-relaunched per user's "take initiative". Watching; if infra
benchmarks fire again and OOM the run, will hold + escalate. Monitoring 10-min until 2 steps, then hourly.
