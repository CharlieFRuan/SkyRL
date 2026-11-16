# Eval settings research (2026-11-04)

## 1. MiMo report: MiMo-V2.6, §7.2, Table 6

Source: "MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement", https://arxiv.org/html/2610.11959v1. §7 is "Open Foundations for Agentic RL".

- Table 6 lists Qwen3.5-9B at **SWE-bench Verified avg@3 = 60.0** and **Terminal Bench 2.1 avg@1 = 27.0**. SFT scores 61.1 / 37.1, RL 66.2 / 52.8. Caption: "Code uses single-harness RL."
- §5.2 "Baseline Configuration": "All baseline models with configurable reasoning effort are evaluated at the highest supported setting (max)."
- Table 7 gives Qwen3.5-9B on SWE-V by harness: mini-swe-agent 60.6, claude code 54.8, codex 48.6, mini-harness1–4 58.4/36.4/54.8/57.8. It does not say which harness produced the 60.0.
- **Not found** in the report or the HF card: the harness for Table 6, timeouts, turn limits, context length, temperature/top_p/top_k, sandbox resources. The HF card only enables thinking (`chat_template_kwargs: {enable_thinking: true}`, SGLang `--reasoning-parser mimo`).
- Related but not the report: Xiaomi `uni-agent` docs (`docs/source/benchmark/inference.md`) list Qwen3.5-9B SWE-V at 53.8 Avg@1 ("100 turns, 64K context") and 65.6 ("200 turns, 128K context").
- Qwen's card recommends temp 1.0, top_p 0.95, top_k 20, presence_penalty 1.5 for thinking mode, or temp 0.6 with presence_penalty 0 for coding.

## 2. Artificial Analysis TB2.1

Source: https://artificialanalysis.ai/methodology/intelligence-benchmarking. TB2.1 is now "Superseded by Terminal-Bench 4.0 in Intelligence Index v4.3".

The page says, verbatim:
- "Terminus 2 agent harness in an E2B sandbox environment, with pass@1 scoring averaged over 3 repeats per task"
- "Maximum 'episodes' … are limited to 250"
- "Per-task agent timeout is set to two hours (7,200 seconds), or the task's own specified timeout where that is longer"
- Changelog: "no token limits"
- Temperature "0.6 for reasoning models (unless another temperature is recommended by the model lab)". Max output tokens: "Maximum output tokens allowed".

**Not found:** sandbox CPU/memory, command timeout, parser, summarization settings.

## 3. NeMo Evaluator TB2.1 recipe (Lightning-3.5)

Recipe: Gym `nemotron_recipes/lightning-3.5/instruct/nemo-evaluator/terminal-bench-2.1.yaml`. It pins Evaluator at `230c8411fff8…`, which requires `harbor>=0.3.0,<0.4.0`. Only **harbor 0.3.0** exists in that range.

NEL builds the Harbor `terminus-2` agent itself (`solvers/harbor.py:_create_agent`) and runs it against its own ECS sandbox, not Harbor's job runner.

- **Sampling:** the proxy's `payload_modifier` injects `temperature 1.0` and `top_p 0.95`, overriding Terminus 2's 0.7 default. It also sends `enable_thinking: true` and `skip_special_tokens: false`. `drop_params` strips `max_tokens`/`max_completion_tokens`, so no output cap is sent. LLM timeout is 3600s.
- **Context:** `model_info={max_input_tokens:262144, max_output_tokens:262144, cost 0}`.
- **Thinking:** the `reasoning` interceptor plus `reasoning_replay` (`mode: think_tags`). Each turn's `reasoning_content` is cached and re-prepended as `<think>…</think>` to that assistant message on the next request.
- **Agent kwargs:**
  - `model_name=openai/<model>`, `api_key`.
  - `api_base` is the proxy URL, with `host.docker.internal` rewritten to `127.0.0.1` because the client runs on the host.
  - Left unset: `max_turns` (default 1e6, effectively unlimited), `parser_name` (json), `enable_summarize` (True), proactive threshold (8000).
- **Timeouts:** `run_timeout 7200` with `timeout_strategy: max` gives `max(7200, task [agent].timeout_sec)`, uncapped, plus up to min(120s, 2%) jitter. No `cmd_timeout` is set.
- **Sandbox:** ECS Fargate at 4096 cpu units (4 vCPU) and 8192 MB, 100 GiB disk, `stateful: true`. These apply uniformly: `task.toml` cpus/memory are ignored (`ecs_fargate.py` ~L1852).
- **Run size:** `repeats: 8`, concurrency 15, 89 tasks at terminal-bench-2-1 commit `c5ee500c`.

**Terminus 2 patches** (in `solvers/harbor.py`):
1. **tmux send-keys:** `_tmux_send_keys` and `_split_key_for_tmux` measure UTF-8 **bytes**, not code points, against the 16,000 limit. Oversized keys are binary-search split into `shlex.quote`d chunks. Harbor HEAD already does this natively.
2. **Context-length reset** (rewrite of `_query_llm`): if the full summary hits `ContextLengthExceededError`, it resets the chat to `[messages[0]]` and calls `reset_response_chain()` before fallback 2. The final fallback is a parseable empty-commands response with `task_complete:false`. Still unfixed in Harbor HEAD.
3. **Minimum unwind:** `_unwind_messages_to_free_tokens` removes at least 10 message pairs.
4. **Token counting:** the count is anchored on the API's `prompt+completion` usage, plus a litellm count of only newer messages.
5. **Error matching:** `_is_context_length_error` also matches vLLM's "model's context length" wording.

## 4. Harbor knobs (HEAD 539c117, v0.25.0)

- **CPU and memory:** `--override-cpus 4 --override-memory-mb 8192` set `EnvironmentConfig.override_cpus` and `override_memory_mb` (`models/trial/config.py` ~L226; also in 0.3.0). `--cpus` and `--memory` are enforcement policies. `--ek` passes environment kwargs.
- **Timeout:**
  - Harbor computes `min(override_timeout_sec or task timeout, max_timeout_sec) × multiplier`, via `--agent-timeout-multiplier` or `--timeout-multiplier` (`trial/trial.py`). There is **no "max" strategy**.
  - Task timeouts at tb2-1 HEAD (7131e43):
    - 900s ×48, 1800s ×17, 3600s ×13, 1200s ×5, 2400s ×2
    - one each of 600s, 750s, 7200s
    - **12000s for build-pov-ray**
  - So set `agents[].override_timeout_sec: 7200` in a job YAML and run build-pov-ray separately without the override.
  - Task resources at that HEAD: 83 tasks use 1 cpu; memory is 2048 MB for 68.
- **Terminus 2 `--ak` keys** (values parsed as JSON or Python literals):
  - Keys with defaults: `max_turns` (1e6), `parser_name` (json), `temperature` (None; 0.7 in 0.3.0), `enable_summarize` (True), `proactive_summarization_threshold` (8000).
  - Other keys: `api_base`, `reasoning_effort`, `model_info`, `interleaved_thinking`, `llm_kwargs`, `llm_call_kwargs`.
  - `model_info` is passed to `litellm.register_model` and sets the context limit.
  - The chat-completions path sends no `max_tokens`; add one with `llm_call_kwargs`.

**AA-style:**

```
-a terminus-2 -m openai/<m> -k 3 --ak max_turns=250 --ak api_base=<url> --ak temperature=0.6 --ak 'model_info={"max_input_tokens":262144,"max_output_tokens":262144,"input_cost_per_token":0,"output_cost_per_token":0}'
```

**NVIDIA-style:** add the cpu/memory overrides, use temperature 1.0, and leave `max_turns` unset.
