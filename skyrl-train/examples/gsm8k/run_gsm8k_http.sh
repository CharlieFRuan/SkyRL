#!/bin/bash
set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.
# Uses HTTP endpoint for generation requests instead of direct engine calls.
#
# This script demonstrates:
# 1. Using SkyRLGymHTTPGenerator which sends HTTP requests to /v1/chat/completions
# 2. Enabling vLLM stats logging to see throughput metrics
#
# Usage:
#   bash examples/gsm8k/run_gsm8k_http.sh
#
# To enable vLLM stats logging:
#   bash examples/gsm8k/run_gsm8k_http.sh generator.enable_vllm_logging_stats=true

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run --isolated --extra $INFERENCE_BACKEND python "$SCRIPT_DIR/main_http.py" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_http" \
  trainer.run_name="gsm8k_http_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_http_ckpt" \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  $@


