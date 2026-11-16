#!/usr/bin/env bash
# Fully-async RL with skycap: Terminus 2 (terminus_think.TerminusThink, NeMo Evaluator's patches) on Harbor
# terminal tasks in Daytona-API sandboxes (Daytona, or Dune Cloud with DAYTONA_API_URL), every model call through
# skycap (token mode). Megatron on POLICY_NUM_NODES nodes, vLLM engines on the others (not colocated). Terminus runs
# on the cluster and drives the sandbox through the provider's exec API, so no relay or exposure is needed.
#
#   TASKS_DIR=<train tasks> EVAL_TASKS_DIR=<eval tasks> MODEL=<checkout> RUN_NAME=<name> \
#     bash examples/train_integrations/harbor_skycap/run_fully_async_terminus.sh [overrides...]
#
# Recipe: DPPO (delta 0.15), prompt_mean loss reduction, fused LM-head logprobs, lr 1e-6, adam betas [0.9, 0.98],
# wd 0.01, no std normalization, zero-variance filter, batch = mini batch = 16 prompts x 16 samples, max staleness
# 2, sticky least-loaded routing, a weight-version cache salt. Training samples at temperature 1.0 / top_p 1.0 (rollout
# and trainer logprobs agree); eval samples with EVAL_SAMPLING (skycap.eval_sampling_overrides; default Qwen3.5's
# recommended 0.6 / 0.95 / top_k 20). Every earlier turn's thinking is kept in the history
# (skycap.thinking_retention=all), so a rollout is one path (two or more after a Terminus summarization; all train).
#
# Knobs (environment): TASKS_DIR, EVAL_TASKS_DIR, MODEL, SERVED_MODEL_NAME, TOKENIZER, RUN_NAME, RUN_DIR,
# POLICY_NUM_NODES, NUM_ENGINES, TP_SIZE, MEGATRON_TP, MAX_MODEL_LEN, MAX_CONCURRENCY, TRAJECTORIES_PER_SECOND,
# MAX_STALENESS_STEPS, GROUP_SIZE, TRAIN_BATCH_SIZE, SKYCAP_SERVERS, TRAIN_PATHS, EVAL_INTERVAL, EVAL_SAMPLES,
# EVAL_BEFORE_TRAIN, CKPT_INTERVAL, MAX_CKPTS, EPOCHS, MAX_TURNS, LR, LOGGER, PROJECT, LABELS_OWNER, KEYS_FILE, VENV.
set -euo pipefail

# REPO: the SkyRL checkout (default: this script's), for running a copy of this script from elsewhere.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
HERE="$REPO/examples/train_integrations/harbor_skycap"
cd "$REPO"

MODEL="${MODEL:?set MODEL to the model checkout}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
TASKS_DIR="${TASKS_DIR:?set TASKS_DIR to the directory of Harbor training tasks}"
EVAL_TASKS_DIR="${EVAL_TASKS_DIR:?set EVAL_TASKS_DIR to the directory of Harbor eval tasks}"
RUN_NAME="${RUN_NAME:-terminus-fully-async}"
RUN_DIR="${RUN_DIR:-/mnt/local_storage/$USER/runs/$RUN_NAME}"
KEYS_FILE="${KEYS_FILE:-$HOME/charlie_keys.env}"
LABELS_OWNER="${LABELS_OWNER:-$USER}"

POLICY_NUM_NODES="${POLICY_NUM_NODES:-1}"
NUM_ENGINES="${NUM_ENGINES:-12}"
TP_SIZE="${TP_SIZE:-2}"
MEGATRON_TP="${MEGATRON_TP:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-16}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-2}"
NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-$((TRAIN_BATCH_SIZE * (MAX_STALENESS_STEPS + 1)))}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-500}"
TRAJECTORIES_PER_SECOND="${TRAJECTORIES_PER_SECOND:-2}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-8}"
TRAIN_PATHS="${TRAIN_PATHS:-all}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_SAMPLES="${EVAL_SAMPLES:-3}"
# (A literal "{...}" default inside ${VAR:-...} ends at its first "}", which left a stray "}" after any EVAL_SAMPLING set
# from outside.)
EVAL_SAMPLING_DEFAULT='{temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.0}'
EVAL_SAMPLING="${EVAL_SAMPLING:-$EVAL_SAMPLING_DEFAULT}"
# TRAIN_ON_TIMEOUTS=true: a trial that hits the agent timeout trains with reward 0 instead of masking its whole group.
TRAIN_ON_TIMEOUTS="${TRAIN_ON_TIMEOUTS:-false}"
CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
MAX_CKPTS="${MAX_CKPTS:-2}"
EPOCHS="${EPOCHS:-3}"
# Terminus episodes per trial (Artificial Analysis' Terminal-Bench setting for Qwen3.5: 250).
MAX_TURNS="${MAX_TURNS:-250}"
# Terminus context summarization (proactive and on a context-length error). false: a trial that fills the context ends
# there (ContextLengthExceededError) and trains with reward 0.
SUMMARIZE="${SUMMARIZE:-true}"

mkdir -p "$RUN_DIR/trials"
if [[ -f "$KEYS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$KEYS_FILE"
  set +a
fi
: "${DAYTONA_API_KEY:?export DAYTONA_API_KEY or put it in KEYS_FILE}"
LOGGER="${LOGGER:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo wandb || echo console)}"

# Backstop per trial, above build + the longest agent (TB2.1 under NVIDIA's timeouts: 12000 s) + verifier.
export SKYCAP_TRIAL_TIMEOUT_SEC="${SKYCAP_TRIAL_TIMEOUT_SEC:-21600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}"
export SKYRL_RAY_PG_TIMEOUT_IN_S="${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

# VENV: a `uv sync --extra megatron --extra harbor --extra skycap`ed venv at the same path on every node; the driver
# and every Ray worker run its python (SKYRL_PY_EXECUTABLE) instead of a per-worker `uv run --isolated`.
if [[ -n "${VENV:-}" ]]; then
  export SKYRL_PY_EXECUTABLE="$VENV/bin/python"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  PYTHON=("$VENV/bin/python")
else
  PYTHON=(uv run --isolated --extra megatron --extra harbor --extra skycap)
fi
echo "==> $RUN_NAME: $(find -L "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) train tasks, batch $TRAIN_BATCH_SIZE x $GROUP_SIZE, staleness $MAX_STALENESS_STEPS, concurrency $MAX_CONCURRENCY, run dir $RUN_DIR"
"${PYTHON[@]}" \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_fully_async \
  data.train_data="['$TASKS_DIR']" \
  data.val_data="['$EVAL_TASKS_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name="${PROJECT:-skycap-terminus-fully-async}" \
  trainer.run_name="$RUN_NAME" \
  trainer.logger="$LOGGER" \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS" \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode="${RESUME_MODE:-latest}" \
  trainer.epochs="$EPOCHS" \
  trainer.eval_before_train="${EVAL_BEFORE_TRAIN:-true}" \
  trainer.eval_interval="$EVAL_INTERVAL" \
  generator.eval_n_samples_per_prompt="$EVAL_SAMPLES" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes="$POLICY_NUM_NODES" \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.policy.megatron_config.tensor_model_parallel_size="$MEGATRON_TP" \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=1 \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.fused_lm_head_logprob=true \
  trainer.use_sample_packing=true \
  trainer.max_tokens_per_microbatch="$MAX_MODEL_LEN" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  trainer.fully_async.sample_full_batch=true \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=dppo \
  trainer.algorithm.dppo.delta_low=0.15 \
  trainer.algorithm.dppo.delta_high=0.15 \
  trainer.algorithm.loss_reduction=prompt_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.eps_clip_low=0.5 \
  trainer.algorithm.eps_clip_high=4 \
  trainer.algorithm.zero_variance_filter=true \
  trainer.algorithm.zero_variance_filter_tol=1e-6 \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.algorithm.temperature=1.0 \
  trainer.policy.optimizer_config.lr="${LR:-1.0e-6}" \
  "trainer.policy.optimizer_config.adam_betas=[0.9,0.98]" \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.n_samples_per_prompt="$GROUP_SIZE" \
  generator.apply_overlong_filtering=false \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.batched=false \
  generator.use_cache_salt=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.min_p=0.0 \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second="$TRAJECTORIES_PER_SECOND" \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  skycap.num_servers="$SKYCAP_SERVERS" \
  skycap.train_paths="$TRAIN_PATHS" \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.ttl="${SKYCAP_TTL:-14400}" \
  skycap.tokenizer="$TOKENIZER" \
  skycap.thinking_retention=all \
  "skycap.eval_sampling_overrides=$EVAL_SAMPLING" \
  skycap.train_on_timeouts=$TRAIN_ON_TIMEOUTS \
  skycap.exposure.type=none \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.agent.name=null \
  harbor_trial_config.agent.import_path=examples.train_integrations.harbor_skycap.terminus_think:TerminusThink \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.enable_summarize="${SUMMARIZE:-true}" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.llm_kwargs.timeout=3600 \
  harbor_trial_config.agent.override_timeout_sec=null \
  harbor_trial_config.environment.type=daytona \
  harbor_trial_config.environment.import_path=examples.train_integrations.harbor_skycap.sandbox_relay.environment:RelayDaytonaEnvironment \
  harbor_trial_config.environment.kwargs.labels.owner="$LABELS_OWNER" \
  harbor_trial_config.environment.kwargs.labels.run="$RUN_NAME" \
  harbor_trial_config.environment.kwargs.ttl_minutes="${TTL_MINUTES:-480}" \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=0 \
  harbor_trial_config.environment.override_cpus=null \
  harbor_trial_config.environment.override_memory_mb=null \
  harbor_trial_config.environment.override_storage_mb=null \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
