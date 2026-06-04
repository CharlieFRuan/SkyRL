set -ex

# Mercor TITO fully async RL training with GLM-4.7-Flash + Harbor + SkyRL
#
# Usage:
#   cd SkyRL-private
#   bash examples/train_integrations/harbor_apex/run_tito_glm47_apex_dev_fully_async.sh

# All the knobs below default to 4x8xH200 setup. For 1x8xB200, see run_tito_glm47_apex_dev_fully_async_1node.sh
# which overrides some of the knobs below.
# For 4x8xH200, see run_tito_glm47_apex_dev_fully_async_4nodes.sh

#-----------------------
# Dataset (dev-1920)
#-----------------------
DATA_DIR="${DATA_DIR:-/home/ray/data/harbor}"
TRAIN_DATA_DIR=$DATA_DIR/apex-agents-dev-1920
EVAL_DATA_DIR=$DATA_DIR/apex-agents-eval-99
TRAIN_DATA="['$TRAIN_DATA_DIR']"
EVAL_DATA="['$EVAL_DATA_DIR']"

#-----------------------
# Directories
#-----------------------
RUN_NAME="${RUN_NAME:-mercor_glm47flash_tito_rl}"
SAVE_DIR="${SAVE_DIR:-/mnt/local_storage}"
TRIALS_DIR="${TRIALS_DIR:-$SAVE_DIR/$RUN_NAME/trials_run_$(date +%m%d_%H%M)}"
EXPORT_PATH="${EXPORT_PATH:-$SAVE_DIR/$RUN_NAME/export}"
CKPT_PATH="${CKPT_PATH:-$SAVE_DIR/$RUN_NAME/ckpt}"
mkdir -p "$TRIALS_DIR" "$EXPORT_PATH" "$CKPT_PATH" 2>/dev/null || true

# Staging dir for S3 checkpointing so we do not burn /tmp space. Only used for S3 checkpointing.
export SKYRL_CHECKPOINT_TMPDIR=${SAVE_DIR}/tmp

#-----------------------
# Model
#-----------------------
MODEL_NAME="zai-org/GLM-4.7-Flash"
SERVED_MODEL_NAME="GLM-4.7-Flash"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-64000}"

#-----------------------
# Training parameters
#-----------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MICRO_FWD_BATCH="${MICRO_FWD_BATCH:-1}"
MICRO_TRAIN_BATCH="${MICRO_TRAIN_BATCH:-1}"

#-----------------
# vLLM parameters
#-----------------
GPU_MEMORY_UTILIZATION=0.85
ENFORCE_EAGER=false

#-----------------------
# Fully async knobs
#-----------------------
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-3}"
# num_parallel_generation_workers = mini_batch * (max_staleness + 1) is the maximum
NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-64}"

#-----------------------
# Infrastructure
# If 1x8xB200: Disaggregated
#   inference: 1 engines x TP=4
#   training: Megatron TP=2, CP=1, EP=4 (tested 48K, stable)
# If 4x8xH200:
#   3 nodes for inference: 6 engines x TP=4
#   1 node for training: Megatron TP=4, CP=2, EP=4 (tested 64K, very conservative and stable)
#-----------------------
INFERENCE_TP=4
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-6}"

POLICY_NUM_GPUS_PER_NODE="${POLICY_NUM_GPUS_PER_NODE:-8}"
POLICY_NUM_NODES="${POLICY_NUM_NODES:-1}"
MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_EP="${MEGATRON_EP:-4}"
MEGATRON_CP="${MEGATRON_CP:-2}"
MEGATRON_PP=1
MEGATRON_ETP=1

# GLM-4.7-Flash supports flash attention (v_head_dim == qk_head_dim + qk_rope_head_dim == 256).
# Most other MLA models (DeepSeek-V3, Moonlight) do NOT support flash attention due to
# mismatched Q/V head dimensions. Use flash_attn=false for those models.
FLASH_ATTN=true

# MoE routing flags (DeepSeek-V3 style: sigmoid scoring with expert bias)
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# CPU optimizer offload. With this we can run 64k on 1x8xB200, without this we cannot even run 16k.
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

#---------------
# Rate limiting
#---------------
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND="${TRAJECTORIES_PER_SECOND:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-300}"

#------------------------
# Algorithms
#------------------------
# Dr. GRPO parameters
LOSS_REDUCTION="seq_mean_token_sum_norm"
GRPO_NORM_BY_STD=false
USE_KL_LOSS=false  # changing this to true will require infra knob changes as well

# NOTE(Charlie): Will follow DDog and not use Dr. GRPO for now
# LOSS_REDUCTION="token_mean_legacy"
# GRPO_NORM_BY_STD=true
# USE_KL_LOSS=false  # changing this to true will require infra knob changes as well

# Optimizer parameters (from DDog for GLM-4.7)
# LR=1.0e-6
# WEIGHT_DECAY=0.1
# MAX_GRAD_NORM=1.0
# ADAM_BETAS="[0.9,0.98]"

# Optimizer parameters (040626)
LR=1.0e-6
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
ADAM_BETAS="[0.9,0.999]"

# Rollout correction parameters
# NOTE(Charlie): Will use GLM loss instead of TIS
# # TIS parameters
# TIS_IMP_RATIO_CAP=2.0
# TIS_TYPE=token
POLICY_LOSS_TYPE="rollout_is"

# This is sampling, but we need to feed it to trainer as well
# TEMPERATURE=0.7
TEMPERATURE=1.0

# DAPO Clip higher
EPS_CLIP_HIGH=0.28
EPS_CLIP_LOW=0.2

# ------------------------------------------------------------
# Actual training script
# ------------------------------------------------------------

uv run --isolated --env-file /home/ray/default/SkyRL-private/.env.ray --extra megatron --extra harbor \
  -m examples.train_integrations.harbor_apex.entrypoints.main_tito_harbor_fully_async \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  harbor_trial_config_file=mercor_tito \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.agent.kwargs.tito_tokenizer_name=$MODEL_NAME \
  harbor_trial_config.agent.kwargs.tool_result_max_chars=21000 \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.llm_kwargs.temperature=$TEMPERATURE \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.export_path=$EXPORT_PATH \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$POLICY_NUM_GPUS_PER_NODE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD_BATCH \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH \
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE \
  trainer.algorithm.temperature=$TEMPERATURE \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.policy.optimizer_config.lr=$LR \
  "trainer.policy.optimizer_config.adam_betas=$ADAM_BETAS" \
  trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY \
  trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.eval_interval=20 \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=5 \
  trainer.max_ckpts_to_keep=5 \
  trainer.epochs=3 \
  trainer.resume_mode=latest \
  trainer.logger=wandb \
  trainer.project_name=mercor-rl \
  trainer.run_name=$RUN_NAME \
  generator.inference_engine.served_model_name=$SERVED_MODEL_NAME \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_TP \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=glm47 \
  generator.inference_engine.engine_init_kwargs.reasoning_parser=glm45 \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.apply_overlong_filtering=false \
  generator.batched=false \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "$@"
