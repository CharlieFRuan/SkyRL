#!/usr/bin/env bash
# Penfever-branch step-wise test run:
# small-batch CodeContest/Qwen3-8B smoke test on 8xH100 with Modal sandboxes.
# Mirrors setup_otagent.md §5 but routes caches/trials to /mnt/local_storage
# and uses the user-approved batch sizes (4/4) + Modal (not Daytona).
set -x

# Driver env from ~/.bashrc (DAYTONA/MODAL/HF/WANDB); Ray workers inherit via the
# patched prepare_runtime_environment() in skyrl_train/utils/utils.py.
set -a
source "$HOME/.bashrc" >/dev/null 2>&1 || true
set +a

# Keep uv cache + HF cache + wandb scratch off the 10 GB ~/default cap.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/local_storage/uv_cache}"
export HF_HOME="${HF_HOME:-/mnt/local_storage/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/local_storage/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-/mnt/local_storage/wandb}"
export HYDRA_FULL_ERROR=1
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WANDB_DIR"

DATA_DIR="$HOME/data/harbor/CodeContests"
NUM_GPUS=8
LOGGER="wandb"
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="sandboxes"

RUN_NAME="tbench_codecontests_qwen3_8b_stepwise_test"
SCRATCH="/mnt/local_storage/$RUN_NAME"
TRIALS_DIR="$SCRATCH/trials"
CKPT_PATH="$SCRATCH/ckpts"
EXPORT_PATH="$SCRATCH/exports"
mkdir -p "$CKPT_PATH" "$TRIALS_DIR"

# Memory optimizations (mirrors PR #1542's intent)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

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
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.step_wise_training=true \
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
  generator.http_endpoint_port=18000 \
  generator.sampling_params.max_generate_length=26000 \
  +generator.engine_init_kwargs.served_model_name=Qwen3-8B \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.55 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="$EXPORT_PATH" \
  "$@"
