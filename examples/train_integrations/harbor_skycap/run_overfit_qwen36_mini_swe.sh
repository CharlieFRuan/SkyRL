#!/usr/bin/env bash
# Overfitting run: synchronous GRPO on 16 SWE tasks with mini-swe-agent in Daytona sandboxes, every model call through
# skycap (token mode) so training sees each rollout's exact tokens. One node, 8 GPUs, Megatron and vLLM colocated.
#
#   RELAY_HOST=relay.example.com TASKS_DIR=/path/to/16/tasks bash examples/train_integrations/harbor_skycap/run_overfit_qwen36_mini_swe.sh
#
# Batch: 16 tasks x 8 samples = 128 trials in flight, one optimizer step per training step. Recipe follows the GLM
# overfit-32 sync run (GRPO, lr 1e-6, eps clip 0.2/0.28, no KL, per-token TIS against the rollout log-probs) with Qwen3.6
# Megatron settings from main's run_megatron_dapo_qwen3.5_35b_a3b.sh (TP 4, EP 8, optimizer offloaded, GDN prefill on
# triton). Each task's sandbox size comes from its own task.toml (sandbox_provision_pass.md).
#
# Knobs (environment): RELAY_HOST, SKYCAP_PORT, NODE_IP, TASKS_DIR, RUN_DIR, EPOCHS (= steps: 16 tasks, batch 16),
# MAX_MODEL_LEN, LABELED_DAYTONA_DIR, KEYS_FILE (DAYTONA_API_KEY, WANDB_API_KEY), LOGGER, RUN_NAME, GROUP_SIZE,
# TRAIN_PATHS (skycap path rule: all or final), CKPT_INTERVAL, R3=1 (rollout routing replay: vLLM returns each
# token's routed experts and Megatron replays them).
#
# Multi-node: start a Ray cluster from the repo's .venv on every node first (same repo path everywhere), then run this on
# the head with NUM_NODES=<n>, RAY_ADDRESS=<head>:<port> and USE_VENV=1 (the driver runs .venv/bin/python, so Ray
# workers on every node use that same interpreter instead of uv's per-job env).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/examples/train_integrations/harbor_skycap"
cd "$REPO"

MODEL="${MODEL:-/mnt/local_storage/charlieruan/hf/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}"
TASKS_DIR="${TASKS_DIR:?set TASKS_DIR to the directory of Harbor tasks to overfit}"
MSWEA_CONFIG="${MSWEA_CONFIG:-$HERE/mini_swe_agent_qwen36_train.yaml}"
MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.6}"
RELAY_HOST="${RELAY_HOST:?set RELAY_HOST to the public host name of the relay}"
SKYCAP_PORT="${SKYCAP_PORT:-11500}"
NODE_IP="${NODE_IP:-}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TRAIN_PATHS="${TRAIN_PATHS:-all}"
CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
NUM_NODES="${NUM_NODES:-1}"
EPOCHS="${EPOCHS:-50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128000}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_ENGINES="${NUM_ENGINES:-$((4 * NUM_NODES))}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_EP="${MEGATRON_EP:-8}"
LR="${LR:-1.0e-6}"
LABELED_DAYTONA_DIR="${LABELED_DAYTONA_DIR:-/home/charlieruan/mimo/rl-env-cleaner/mimo_swe}"
KEYS_FILE="${KEYS_FILE:-$HOME/charlie_keys.env}"
RUN_NAME="${RUN_NAME:-qwen36-overfit16-mini-swe-sync}"
RUN_DIR="${RUN_DIR:-/mnt/local_storage/$USER/runs/$RUN_NAME}"
mkdir -p "$RUN_DIR/trials"

NUM_PROMPTS="$(find -L "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[[ "$NUM_PROMPTS" -gt 0 ]] || { echo "no tasks in $TASKS_DIR" >&2; exit 1; }
[[ -f "$LABELED_DAYTONA_DIR/labeled_daytona.py" ]] || { echo "no labeled_daytona.py in $LABELED_DAYTONA_DIR" >&2; exit 1; }
if [[ -f "$KEYS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$KEYS_FILE"
  set +a
fi
: "${DAYTONA_API_KEY:?export DAYTONA_API_KEY or put it in KEYS_FILE}"
LOGGER="${LOGGER:-$([[ -n "${WANDB_API_KEY:-}" ]] && echo wandb || echo console)}"

export PYTHONPATH="$LABELED_DAYTONA_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SKYRL_PYTHONPATH_EXPORT=1
# Backstop for a wedged sandbox (agent + verifier + build timeouts fit well inside it).
export SKYCAP_TRIAL_TIMEOUT_SEC="${SKYCAP_TRIAL_TIMEOUT_SEC:-7200}"
export MIMO_RUN_ID="${MIMO_RUN_ID:-$RUN_NAME}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

EXPOSURE_ARGS=(
  skycap.exposure.type=examples.train_integrations.harbor_skycap.relay_exposure:RelayPathExposure
  skycap.exposure.kwargs.relay_host="$RELAY_HOST"
  skycap.exposure.kwargs.port="$SKYCAP_PORT"
)
if [[ -n "$NODE_IP" ]]; then
  EXPOSURE_ARGS+=(skycap.exposure.kwargs.node_ip="$NODE_IP")
fi

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=$((32 * 1024 * 1024 * 1024))
export RAY_enable_ray_event=0
export RAY_task_events_report_interval_ms=0
unset RAY_enable_core_worker_ray_event_to_aggregator RAY_enable_export_api_write_config
unset RAY_DASHBOARD_AGGREGATOR_AGENT_EVENTS_EXPORT_ADDR RAY_DASHBOARD_AGGREGATOR_AGENT_PUBLISHER_MAX_RETRIES

echo "==> $RUN_NAME: $NUM_PROMPTS tasks x $GROUP_SIZE samples, $EPOCHS steps, run dir $RUN_DIR"
R3_ARGS=()
if [[ "${R3:-0}" == 1 ]]; then
  R3_ARGS=(
    trainer.policy.megatron_config.moe_enable_routing_replay=true
    generator.inference_engine.enable_return_routed_experts=true
  )
fi
if [[ "${USE_VENV:-0}" == 1 ]]; then
  PYTHON=("$REPO/.venv/bin/python")
else
  PYTHON=(uv run --isolated --extra megatron --extra harbor --extra skycap)
fi
"${PYTHON[@]}" -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \
  data.train_data="['$TASKS_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=skycap-overfit \
  trainer.run_name="$RUN_NAME" \
  trainer.logger="$LOGGER" \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.ckpt_at_epoch_end=false \
  trainer.max_ckpts_to_keep=2 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode="${RESUME_MODE:-latest}" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes="$NUM_NODES" \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.policy.megatron_config.tensor_model_parallel_size="$MEGATRON_TP" \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size="$MEGATRON_EP" \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
  trainer.use_sample_packing=true \
  trainer.max_tokens_per_microbatch="$MAX_MODEL_LEN" \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.epochs="$EPOCHS" \
  trainer.train_batch_size="$NUM_PROMPTS" \
  trainer.policy_mini_batch_size="$NUM_PROMPTS" \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.policy.optimizer_config.lr="$LR" \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.algorithm.temperature=1.0 \
  generator.n_samples_per_prompt="$GROUP_SIZE" \
  generator.apply_overlong_filtering=true \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.batched=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.min_p=0.0 \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=2 \
  generator.rate_limit.max_concurrency=$((NUM_PROMPTS * GROUP_SIZE)) \
  skycap.num_servers="$SKYCAP_SERVERS" \
  skycap.train_paths="$TRAIN_PATHS" \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.renderer=qwen3.6 \
  skycap.chat_template_kwargs.enable_thinking=true \
  skycap.chat_template_kwargs.preserve_thinking=true \
  skycap.thinking_retention=all \
  "${EXPOSURE_ARGS[@]}" \
  "${R3_ARGS[@]}" \
  harbor_trial_config.trials_dir="$RUN_DIR/trials" \
  harbor_trial_config.agent.name=mini-swe-agent \
  harbor_trial_config.agent.kwargs.version="$MINI_SWE_AGENT_VERSION" \
  harbor_trial_config.agent.kwargs.config_file="$MSWEA_CONFIG" \
  harbor_trial_config.environment.type=daytona \
  harbor_trial_config.environment.import_path=labeled_daytona:LabeledDaytonaEnv \
  harbor_trial_config.environment.override_cpus=null \
  harbor_trial_config.environment.override_memory_mb=null \
  harbor_trial_config.environment.override_storage_mb=null \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=0 \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"
