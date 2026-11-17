#!/usr/bin/env bash
# Synchronous overfit run (the recipe of Qwen3.6 run vci3ga5w on the GCP B200 cluster: colocated Megatron + vLLM, GRPO
# with std norm, token_mean, clip 0.2/0.28, token TIS cap 2, all paths, overlong filtering, agent timeout 1200 s) on the
# 16-task `overfit16fast` subset, group 8. Run ON a GPU worker; the driver joins the Anyscale Ray cluster.
#
#   LR=1.0e-6 bash ops/run_overfit.sh <run-name>     (env: MODEL_FAMILY, NUM_NODES (2), EPOCHS (20 = steps), LR, ...;
#                                                     see run_overfit_qwen36_mini_swe.sh)
set -euo pipefail
D=/mnt/local_storage/charlie
export RUN_NAME=${1:?run name}
export RAY_ADDRESS=auto
export MODEL_FAMILY=${MODEL_FAMILY:-nemotron-3.5}
if [[ "$MODEL_FAMILY" == nemotron-3.5 ]]; then
  export MODEL=${MODEL:-$D/hf/Nemotron-3.5-Lightning-30B-A3B} SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-nemotron-3.5-lightning}
else
  export MODEL=${MODEL:-$D/hf/Qwen3.6-35B-A3B}
fi
export TASKS_DIR=${TASKS_DIR:-$D/data/overfit16fast}
export NUM_NODES=${NUM_NODES:-2} GROUP_SIZE=${GROUP_SIZE:-8} TRAIN_PATHS=${TRAIN_PATHS:-all} EPOCHS=${EPOCHS:-20}
export SANDBOX_RELAY=1
export LABELED_DAYTONA_DIR=$D/rl-env-cleaner/mimo_swe
export KEYS_FILE=$D/keys.env
export RUN_DIR=$D/runs/$RUN_NAME
export XDG_CACHE_HOME=$D/.cache
export UV_CACHE_DIR=/home/ray/.cache/uv
export VENV=${VENV-$D/venvs/megatron}
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai} WANDB_ENTITY=${WANDB_ENTITY:-sky-posttraining-uc-berkeley}
# Megatron + AWS EFA, as in run_fully_async_qwen36_mini_swe.sh.
export CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600} SKYRL_RAY_PG_TIMEOUT_IN_S=${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}
export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 NCCL_DEBUG=WARN
ulimit -n 524288 || true
mkdir -p "$RUN_DIR"
cd $D/SkyRL-skycap
exec bash examples/train_integrations/harbor_skycap/run_overfit_qwen36_mini_swe.sh "${@:2}"
