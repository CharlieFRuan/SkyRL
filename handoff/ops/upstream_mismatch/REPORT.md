# Nemotron-3.5-Lightning: vLLM-vs-Megatron logprob mismatch (root cause, fix, upstreaming)

Status: 2026-11-04. Mainline A/B done (section 5): mainline already fixes it for Megatron.

W&B entity for every link: `sky-posttraining-uc-berkeley` (https://wandb.ai/sky-posttraining-uc-berkeley).
Metric everywhere unless noted: SkyRL's `policy/rollout_train_logprobs_abs_diff_mean`, the token mean of
|vLLM rollout logprob - Megatron logprob| over response tokens. W&B also logs it as
`policy/minibatch_rollout_logprobs_abs_diff_mean`.

## 1. TL;DR

- **Symptom.** Without R3, Nemotron-3.5-Lightning's mismatch was about 3x Qwen3.6's on the same recipes:
  - GSM8K: 0.045-0.053.
  - Long agentic SkyCap overfits: 0.072.
- **Root cause.** The routers' `e_score_correction_bias` was rounded to bf16 by the weight sync.
  - This bias is the only fp32 tensor in the checkpoint. It only chooses experts.
  - Its values sit at about 37-47 with a spread of about 0.1 per layer. bf16's step there is 0.25, so each layer's bias
    collapses to about one value.
  - vLLM loads the correct fp32 bias at startup. SkyRL's first weight sync, before step 1, then overwrote it with the
    collapsed bf16 one.
  - Result: every rollout routed without the correction, while Megatron (fp32 bias) routed with it.
- **Why R3 hid it.** R3 makes Megatron replay vLLM's choices, so both sides route without the bias (0.008). That
  removes the gap but trains a degraded router.
- **Fix.** Send the bias to vLLM in fp32.
  - Mine was suffix-based (`e_score_correction_bias`), covered Megatron and FSDP, and is uncommitted in the SkyCap tree.
  - The fix measured on GSM8K without R3: 0.045-0.053 -> 0.020. That is close to NeMo-RL (0.014), prime-rl (0.010)
    and Qwen3.6 without R3 (0.016).
- **Mainline status.** Mainline merged an equivalent, more general fix for the Megatron path on 2026-10-26.
  - It is **#2543** (`[weight sync] Keep fp32 and integer params in their dtype on the wire`, by avigyabb), with a
    follow-up **#2618** (2026-11-04) for the sharded-RDT path.
  - Rule: every fp32 tensor, plus integer tensors, keeps its dtype on the wire.
  - That PR was measured on DeepSeek-V4, not Nemotron.
  - Section 5 re-runs the Nemotron A/B on clean mainline to confirm it covers this case.
- **What remains to upstream** is in section 6.

## 2. Where the fixes are

| What | Where | Status |
|---|---|---|
| My original fix (Megatron + FSDP sources) | `~/hackskyrl/src/SkyRL-skycap`, branch `skycap-mini-swe` (fork `CharlieFRuan/SkyRL`), `skyrl/backends/skyrl_train/weight_sync/sources.py`: `FP32_SYNC_SUFFIXES = ("e_score_correction_bias",)` + `wire_dtype(name, dtype)`, used in `FsdpWeightSource.metadata/__iter__` and `MegatronWeightSource.metadata/__iter__` | uncommitted |
| Unit tests for it | same tree, `tests/backends/skyrl_train/weight_sync/test_sources.py`: `TestMegatronWeightSource.test_router_score_bias_stays_float32`, `TestFsdpRouterScoreBias` | uncommitted |
| Load-time diagnostic | same tree, `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py` `_log_expert_bias` (`SKYRL_DEBUG_EXPERT_BIAS=1`: logs the loaded routers' `expert_bias` dtype and spread) | uncommitted, debug only |
| Mainline fix (Megatron) | NovaSky-AI/SkyRL `ef021cd5` (#2543, 2026-10-26): `MegatronWeightSource._wire_dtype` keeps fp32/integer tensors' dtype | merged |
| Mainline fix (sharded RDT) | NovaSky-AI/SkyRL `3ba6f33d` (#2618, 2026-11-04): module-level `wire_dtype(tensor, dtype)` in `sources.py`, used by `sharded_rdt/rdt_send.py` | merged |
| Mainline FSDP path | `FsdpWeightSource` still casts every tensor to the sync dtype | **not fixed** (see section 6) |

## 3. Experiments that found the cause (SkyRL fork, Oct 15-16)

### 3.1 GSM8K, SkyRL main's Lightning example settings, one 8xB200 node, steps 1-3

All runs are in project `nemotron_3_5_lightning_30b_grpo`. Setup: Megatron TP4/EP8 with 2 colocated vLLM engines x TP4.
Launcher: `ops/run_gsm8k_nemotron.sh`.

| Run | What changed | mean \|dlogprob\| | max | W&B |
|---|---|---|---|---|
| `nem35-gsm8k-base-1015-1402` | baseline (CUDA graphs) | 0.045 / 0.053 / 0.052 | 12-14 | [fc1kdvsd](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/fc1kdvsd) |
| `nem35-gsm8k-eager-1015-1402` | vLLM eager | 0.045 / 0.052 / 0.051 | 9-12 | [jwcua85v](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/jwcua85v) |
| `nem35-gsm8k-tp2x4-1015-1402` | 4 engines x TP2 | 0.046 / 0.053 / 0.052 | 10-16 | [2t9q7ymo](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/2t9q7ymo) |
| `nem35-gsm8k-r3-1015-1402` | R3 (routing replay) | 0.008 / 0.008 / 0.008 | 1.1-1.9 | [cfvtwb61](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/cfvtwb61) |
| `nem35-gsm8k-cutlass-1016-0050` | vLLM `moe_backend=flashinfer_cutlass` (NeMo-RL's) | 0.046 / 0.053 | 16 | [26k41bz3](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/26k41bz3) |
| `nem35-gsm8k-flashattn-1016-0115` | vLLM `attention_backend=FLASH_ATTN` (NeMo-RL's) | 0.043 (step 1) | 12 | [hbt7g9ks](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/hbt7g9ks) |
| `nem35-gsm8k-biasdiag-1016-0145` | `SKYRL_DEBUG_EXPERT_BIAS=1` load diagnostic (Megatron's bias is fp32, exact match with the checkpoint) | 0.042 (step 1) | 39 | [mu2ux3gy](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/mu2ux3gy) |
| **`nem35-gsm8k-fp32bias-1016-0155`** | **the fix, no R3** | **0.0197 / 0.0205** | 9.6 / 5.4 | [63aqqnsp](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/63aqqnsp) |
| `nem35-gsm8k-fp32bias-r3-1016-0155` | the fix + R3 | 0.0075 / 0.0075 | 1.1 | [z7koy5o1](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/z7koy5o1) |

What the table rules out:
- CUDA graphs, inference TP, the MoE kernel and the attention backend.
- Not shown in the table: the vLLM mamba SSM cache dtype (already fp32) and the Megatron bias load (exact).

Only R3 and the bias fix move the number.

### 3.2 Long agentic trajectories (SkyCap, overfit16fast, about 22k tokens per trajectory, step 1)

| Run | Setting | mean \|dlogprob\| | W&B (project `skycap-overfit`) |
|---|---|---|---|
| `nem35-of16fast-g8-all-lr1e-6-1015-1249` | no R3, bf16 bias sync | 0.072 | [s29odotd](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/s29odotd) |
| `nem35-of16fast-g8-all-lr3e-6-1015-1249` | no R3, bf16 bias sync | 0.071 | [tcz31f88](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/tcz31f88) |
| `nem35-of16fast-g8-all-r3-lr1e-6-1015-1430` | R3 | 0.0103 | [f50x3z2w](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/f50x3z2w) |
| `nem35-of16fast-g8-all-r3-lr3e-6-1015-1430` | R3 | 0.011 | [flcc3asj](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/flcc3asj) |
| `nem35-of16v2-g8-all-nor3-lr5e-6-4node-1016-0441` | **fix, no R3** | 0.022-0.030 | [gxj7q4gh](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/gxj7q4gh) |
| Qwen3.6 reference, `qwen36-overfit16fast-g8-allpaths-lr5e-6` | no R3 | 0.0157 | [vci3ga5w](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-overfit/runs/vci3ga5w) |

Two more checks:
- **Not SkyCap's bookkeeping.** I re-scored 12 recorded trajectories with a fresh vLLM server's prefill
  (`ops/rescore_skycap.py`). That gave 0.074, the same as Megatron's 0.072. A fresh server loads the correct fp32 bias;
  the recorded rollouts came from the bias-collapsed engine.
- **Other runs with the fix, no R3:**
  - The DAPO run (`nem35-dapo-4node-nor3-1016-0147`, project `nemotron_3_5_lightning_dapo`,
    [dxnz554x](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_dapo/runs/dxnz554x)) held
    0.012-0.017.
  - The 218-step fully-async run (`nem35-mimo-fa-1016-0654`, project `skycap-mimo-fully-async`) held 0.017-0.023 the
    whole time. W&B: [5doiopns](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/5doiopns)
    (steps 1-136) and [uc73dejf](https://wandb.ai/sky-posttraining-uc-berkeley/skycap-mimo-fully-async/runs/uc73dejf)
    (resume to step 218).

## 4. Other RL frameworks (GSM8K, no R3, same model, one 8xB200 node, Oct 16)

All in project `nemotron-mismatch-xfw`. Each framework's metric was patched to also log SkyRL's definition: the token
mean of |d| over all response tokens.

| Framework | Setup | mean \|dlogprob\| (SkyRL def.) | W&B |
|---|---|---|---|
| **NeMo-RL** (NVIDIA-NeMo/RL `cf63b8a3`, v0.8.0, vLLM 0.29) | Megatron TP4+SP/EP8, colocated vLLM 2xTP4, NVIDIA's Lightning vLLM preset, router frozen fp32, no IS correction | **0.0135 / 0.0142 / 0.0136** (steps 1-3), max 1.8-4.1, p99 ~0.19 | [fi503cm5](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron-mismatch-xfw/runs/fi503cm5) |
| **prime-rl** (`0956a885`, v0.10.0, vLLM 0.31) | own FSDP NemotronH trainer EP4 + 4 vLLM replicas, TRITON MoE backend | **0.010-0.013** (k3 KL ~0.0017) | [2ddef690595c47b2b08557ada68e5432](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron-mismatch-xfw/runs/2ddef690595c47b2b08557ada68e5432) |
| Miles (radixark/miles `871cc7d4`, SGLang) | Megatron TP4/EP8 and Miles' verified TP2/PP2/EP2 | ~1.28: **invalid setup** (its trainer is not computing the same model, likely the experimental Lightning bridge shim; not debugged) | [6opyrc3d](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron-mismatch-xfw/runs/6opyrc3d), [9lygqksw](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron-mismatch-xfw/runs/9lygqksw) (first try [gock7e5n](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron-mismatch-xfw/runs/gock7e5n) crashed in TE cuDNN attention) |
| SkyRL before the fix | section 3.1 | 0.045-0.053 | fc1kdvsd |
| SkyRL with the fix | section 3.1 | 0.020 | 63aqqnsp |

What this shows:
- NeMo-RL and prime-rl both send the bias in fp32, and both sit at 0.010-0.014. So the model isn't inherently
  mismatched; SkyRL's sync was the outlier.
- **NVIDIA's own Lightning RL recipe doesn't use R3** (NeMo-RL `examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml`).
  It uses TIS clamped to [0.2, 5] and drops whole sequences with mean exp|d| > 2.
- All of Miles' MoE recipes enable R3.

Possible sources of the small remaining gap after the fix (0.020 vs NeMo-RL's 0.014), none tested in isolation:
- NeMo-RL's fp32 logits (`defer_fp32_logits`) and its optional NemotronH fp32 LM-head patch.
- Prompt and length differences.

Framework launch scripts: `ops/miles/`. The prime-rl and NeMo-RL ones were on worker disks, which have been wiped.

## 5. Mainline re-test (2026-11-04): clean NovaSky-AI/SkyRL clone, one 8xB200 node, GSM8K, no R3, 10 steps

- **Code:** mainline `f854d6d9` (2026-11-04, vLLM 0.33.0, torch 2.13). The clone is at `~/hackskyrl/src/SkyRL-upstream`.
- **Launcher:** `ops/upstream_mismatch/run_gsm8k_ab.sh`. It uses mainline's Lightning example hyperparameters and the
  8-GPU mesh from its comments (TP4/EP8, 2 vLLM engines x TP4). A private Ray head pins it to the node.
- **Arms:**
  - **before:** mainline + `ops/upstream_mismatch/before_bf16_wire.patch`. `wire_dtype` returns the sync dtype for
    everything, which is the pre-#2543 behavior.
  - **after:** mainline as is.

Per step, mean |dlogprob| (max):

| Step | before (#2543 disabled) | after (mainline as is) |
|---|---|---|
| 1 | 0.0448 (36.7) | 0.0199 (35.7) |
| 2 | 0.0535 (22.9) | 0.0205 (6.8) |
| 3 | 0.0522 (15.5) | 0.0199 (5.2) |
| 4 | 0.0557 (22.4) | 0.0208 (7.4) |
| 5 | 0.0572 (10.0) | 0.0206 (7.3) |
| 6 | 0.0617 (16.5) | 0.0201 (7.7) |
| 7 | 0.0620 (11.8) | 0.0201 (7.3) |
| 8 | 0.0661 (17.6) | 0.0209 (5.3) |
| 9 | 0.0708 (15.0) | 0.0219 (14.4) |
| 10 | 0.0720 (14.6) | 0.0214 (7.6) |

Runs (project `nemotron_3_5_lightning_30b_grpo`):
- before: `nem35-main-before-1104` ([5xr52ubp](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/5xr52ubp)), node <node>.
- after: `nem35-main-after-1104` ([c5jy53xs](https://wandb.ai/sky-posttraining-uc-berkeley/nemotron_3_5_lightning_30b_grpo/runs/c5jy53xs)), node <node>.

Reading the numbers:
- **after:** flat at 0.020-0.022 for all 10 steps. That matches my fork's fix (0.0197 / 0.0205, 63aqqnsp).
- **before:** the October level at step 1 (0.045), then it climbs every step, to 0.072 at step 10.
- The rewards track each other (0.68-0.88 in both), which is expected over 10 steps.

The climb is consistent with the router bias changing during training, while the before arm keeps sending vLLM a
collapsed bf16 copy. I haven't verified it, but a likely mechanism exists:
- Megatron's `moe_router_enable_expert_bias` does aux-loss-free load balancing, which updates `expert_bias` every step
  (`moe_router_bias_update_rate`).
- NeMo-RL freezes the router for RL.

## 6. What is left to upstream

**Code for Nemotron on Megatron: nothing.** #2543 (merged 2026-10-26) fixes it on today's mainline, and section 5 shows
it.

Optional follow-ups:

1. **Docs (`docs/content/docs/models/nemotron.mdx`).**
   - It says "vLLM vs Megatron logprobs differ by ~0.01, same as other models". That is likely the pre-sync CI number.
   - In RL (after weight syncs) we measure ~0.020 on GSM8K and 0.020-0.03 on long agentic rollouts, without R3.
   - A short note that R3 is not needed once fp32 params sync in fp32 would help users.
2. **Regression test.** A Nemotron-named unit test: `e_score_correction_bias` keeps fp32 on the wire. #2618's tests
   cover fp32 tensors generically, so this one is optional.
3. **Router bias updates during RL.** Should RL disable the expert-bias update (`moe_router_bias_update_rate=0`), as
   NeMo-RL freezes the router? It deserves its own check (see the "before" climb).
4. **FSDP (`FsdpWeightSource`)** still casts every tensor, including `e_score_correction_bias`, to the sync dtype. We
   decided to skip it: Nemotron-3.5 in SkyRL is Megatron-only.
