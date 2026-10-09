# SkyCap parity check: SWE-bench Verified 100, mini-swe-agent, Qwen3.6-35B-A3B (2026-10-09)

**Question:** does running mini-swe-agent through SkyCap (token mode) score the same as stock Harbor against a plain vLLM
endpoint?

**Answer:** yes, after two SkyCap fixes found by this check.

## Setup (identical in both arms)

- **Tasks:** Marin's 100-task subset, `DCAgent2/swebench-verified-random-100-folders` at revision `0c553bd6`.
  - Sandboxes: Daytona, 1 vCPU / 4 GB / 10 GB, ephemeral `LabeledDaytonaEnv`.
- **Agent:** mini-swe-agent 2.4.6, config `mini_swe_agent_qwen36.yaml`.
  - Unlimited steps; agent 1800 s, verifier 300 s.
  - 73,728-token context, 8,192 output tokens per turn.
- **Sampling:** temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0.
  - Thinking on and kept across turns (`preserve_thinking` / `thinking_retention=all`).
- **Engines:** vLLM 0.30.0 from SkyRL main's locked env; 4 engines × TP 2 on one 8×B200 node per arm.
  - Parsers: qwen3 reasoning, qwen3_coder tools.
- **Arms:**
  - **Baseline:** `run_swebench_baseline_mini_swe.sh`, stock `harbor run` → https relay `/base` → affinity router → vLLM.
  - **SkyCap:** `run_swebench_eval_mini_swe.sh`, the SkyRL generator → SkyCap token mode, reached by the sandboxes through
    the relay (`relay_exposure.RelayPathExposure`) → SkyRL's inference router → vLLM.
    - `max_tokens_policy=reject` (refuse prompt + max_tokens > max_model_len, as vLLM does).
    - One attempt per trial (`SKYCAP_TRIAL_ATTEMPTS=1`).
- **Reference:** Marin publishes Qwen3.6-35B-A3B at 72.2 on this subset (Terminus-2, 1 trial, n=97). Qwen's model card
  reports 73.4 on the full Verified set with an internal scaffold. mini-swe-agent lands a few points lower.

## Results (6 runs of 100 tasks per arm; `compare_parity.py --split -1`)

| Arm | Mean over runs | sd | Turns p50 | Context-limit exits |
|---|---|---|---|---|
| Baseline (stock Harbor + vLLM) | **0.673** | 0.027 | 64 | 17.2% |
| SkyCap v1 (3 runs) | 0.637 | 0.035 | 61 | 14.7% |
| SkyCap v2: fix 1 | 0.647 | 0.030 | 64 | 16.3% |
| **SkyCap v3: fixes 1 + 2** | **0.663** | 0.027 | 62 | 14.7% |

SkyCap v3 − baseline, paired per task: **−0.010 (95% CI −0.043 to +0.023)**. Peak prompt and output token distributions
also match (p50/p90 peak prompt 39.2k/64.9k vs 40.5k/65.0k).

## Bugs found and fixed

1. **Tool calls without `<tool_call>`.** Qwen3.6 sometimes emits `<function=bash>…</function></tool_call>` with no opening
   tag.
   - vLLM's qwen3_coder parser still extracts the call; SkyCap's `renderers` parser didn't.
   - The agent got "No tool calls found" errors in 43 of 300 v1 trajectories (5 gave up with RepeatedFormatError), against
     1 of 600 in the baseline.
   - Fix: `RenderersRenderer.parse` falls back to vLLM-style parsing of bare `<function=…>` blocks (commit 8fb51162).
2. **Tool definitions rendered differently.** SkyCap flattened tools to the function object before rendering, so every
   prompt's `<tools>` block read `{"name": …}` where Qwen's template (and vLLM) prints the OpenAI form
   `{"type": "function", "function": {…}}`.
   - Fix: wrapped tools for the Qwen3.5/3.6 renderers (commit 49b3d9d4).
   - After it, SkyCap's prompts are token-identical to the HF template on real trajectories, through 40 turns and 42k
     tokens.

## Other parity-relevant changes

- **`max_tokens_policy=reject`:** SkyCap used to clamp max_tokens to the room left; vLLM 0.30 rejects the request instead.
- **`SKYCAP_TRIAL_ATTEMPTS`:** one attempt in eval, like `harbor run`.
- **`SKYCAP_TRIAL_TIMEOUT_SEC`:** a hard per-attempt backstop, after a wedged Daytona sandbox hung a trial past every
  Harbor timeout.
- **Engine env:** the baseline vLLM must come from SkyRL's locked env, with the flashinfer jit-cache/cubin wheels. A plain
  `pip install vllm==0.30.0` JIT-compiles the trtllm MoE kernels for tens of minutes.

## Terminal-Bench (TB-Lite and TB 2.1), 2026-10-09

**Same check, same agent** (mini-swe-agent 2.4.6), with `mini_swe_agent_qwen36_tb.yaml`:
- 131,072-token context and 16,384 output tokens per turn (the training settings).
- Each task's own agent and verifier timeouts and sandbox sizes.
- 3 runs per arm.

**Tasks:**
- **TB-Lite:** Harbor registry `openthoughts-tblite` 2.0 (OpenThoughts-TBLite @ `f075e463`), 100 tasks.
  - `sympy-bug-fix` dropped: its verifier passes with a nop agent.
  - 98 tasks are scored in both arms.
- **TB 2.1:** `harborframework/terminal-bench-2.1` @ `3e235dff` on Hugging Face, 89 tasks.

| Dataset | Baseline (stock Harbor + vLLM) | SkyCap | SkyCap − baseline, paired (95% CI) |
|---|---|---|---|
| TB-Lite | 0.502 (sd 0.039) | 0.498 (sd 0.016) | −0.004 (−0.066 to +0.060) |
| TB 2.1 | 0.395 (sd 0.020) | 0.375 (sd 0.017) | −0.021 (−0.079 to +0.036) |

**Behaviour matches too:**
- Turns: p50 22 vs 25 (TB-Lite), 31 vs 31 (TB 2.1).
- Peak prompt p90: 85k vs 81k, and 113k vs 112k.
- Both arms hit the same 114.7k ceiling (131,072 − 16,384), so `max_tokens_policy=reject` matches vLLM.

**Not comparable to published numbers:** the TB 2.0 model-card score (51.5) used Terminus-2 with 256k context, 80k output
tokens and 32 CPU / 48 GB sandboxes.

**Unbridged calls (19 of 564 TB trials, 1–2 calls each)** follow a reply that hit the 16k per-turn cap
(`finish_reason: length`, cut off mid-thinking).
- The agent replays that truncated reply, which renders closed (`</think>`, `<|im_end|>`) and so can't extend the sampled
  tokens. SkyCap falls back to a fresh render of the messages, which is exactly what vLLM does on every call.
- No effect on the eval. In training, the trajectory's graph branches at that call (`train_paths=all` trains both paths).
