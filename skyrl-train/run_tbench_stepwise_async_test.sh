#!/usr/bin/env bash
# Penfever-branch fully-async step-wise test run:
# validates PR #1536 by flipping `colocate_all=false`, which flips main_tbench's
# trainer selection to FullyAsyncRayPPOTrainer. Uses 4 GPUs for FSDP policy/ref
# and 4 for inference engines.
set -x

set -a
source "$HOME/.bashrc" >/dev/null 2>&1 || true
set +a

export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/local_storage/uv_cache}"
export HF_HOME="${HF_HOME:-/mnt/local_storage/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/local_storage/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-/mnt/local_storage/wandb}"
export HYDRA_FULL_ERROR=1
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WANDB_DIR"

DATA_DIR="$HOME/data/harbor/CodeContests"
POLICY_GPUS=4
INFERENCE_ENGINES=4
LOGGER="wandb"
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="sandboxes"

RUN_NAME="tbench_codecontests_qwen3_8b_stepwise_async_test"
SCRATCH="/mnt/local_storage/$RUN_NAME"
TRIALS_DIR="$SCRATCH/trials"
CKPT_PATH="$SCRATCH/ckpts"
EXPORT_PATH="$SCRATCH/exports"
mkdir -p "$CKPT_PATH" "$TRIALS_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# NOTE: `colocate_all=false` switches main_tbench to FullyAsyncRayPPOTrainer.
# Policy/ref share 4 GPUs (FSDP), inference engines get the other 4.
# num_parallel_generation_workers must fit the small-batch regime:
# with policy_mini_batch_size=4 and max_staleness_steps=1 we allow at most
# `policy_mini_batch_size * (max_staleness_steps + 1) = 8` in-flight groups.
exec uv run --isolated --extra vllm --extra sandboxes \
  --with "./sandboxes" \
  --with "daytona>=0.164.0" \
  --with "socksio" \
  -m examples.terminal_bench.entrypoints.main_tbench \
  data.train_data="['$DATA_DIR']" \
  data.val_data="['$DATA_DIR']" \
  hydra.searchpath=[file://$TBENCH_CONFIG_DIR] \
  +terminal_bench_config=terminal_bench \
  +terminal_bench_config.trials_dir=$TRIALS_DIR \
  +terminal_bench_config.sandboxes_dir=$SANDBOXES_DIR \
  +terminal_bench_config.harbor.max_episodes=16 \
  +terminal_bench_config.harbor.enable_summarize=false \
  +terminal_bench_config.harbor.environment_type=modal \
  +terminal_bench_config.harbor.collect_rollout_details=true \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$POLICY_GPUS \
  trainer.placement.ref_num_gpus_per_node=$POLICY_GPUS \
  generator.num_inference_engines=$INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.step_wise_training=true \
  trainer.fully_async.max_staleness_steps=1 \
  trainer.fully_async.num_parallel_generation_workers=8 \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=null \
  trainer.max_prompt_length=4096 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=18001 \
  generator.sampling_params.max_generate_length=26000 \
  +generator.engine_init_kwargs.served_model_name=Qwen3-8B \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.merge_stepwise_output=true \
  generator.gpu_memory_utilization=0.75 \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="$EXPORT_PATH" \
  "$@"
