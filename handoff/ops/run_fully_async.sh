#!/usr/bin/env bash
# Fully-async RL run (Megatron, 2 training + 2 inference nodes) on MiMo minus exclusions, SWE-bench-100 eval every 20
# steps and before training. Run ON a GPU worker; the driver joins the Anyscale Ray cluster (never `ray stop` it).
#
#   bash ops/run_fully_async.sh <run-name>       (env overrides: see run_fully_async_qwen36_mini_swe.sh; MODEL_FAMILY=
#                                                 qwen3.6 | nemotron-3.5 picks the model, TASKS_DIR the training tasks)
#
# Disk: Megatron checkpoints (~0.5 TB each with optimizer state) land on the training nodes' /mnt/local_storage
# under runs/<run>/ckpts; ckpt every 5 steps, keep 2.
set -euo pipefail
D=/mnt/local_storage/charlie
export RUN_NAME=${1:?run name}
export RAY_ADDRESS=auto
export MODEL_FAMILY=${MODEL_FAMILY:-qwen3.6}
if [[ "$MODEL_FAMILY" == nemotron-3.5 ]]; then
  # The 2026-11 checkout of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (the Oct runs: .../Nemotron-3.5-Lightning-30B-A3B).
  export MODEL=${MODEL:-$D/hf/Nemotron-3.5-Lightning-BF16} SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-nemotron-3.5-lightning}
else
  export MODEL=${MODEL:-$D/hf/Qwen3.6-35B-A3B}
fi
export TASKS_DIR=${TASKS_DIR:-$D/data/mimo-train}
export EVAL_TASKS_DIR=${EVAL_TASKS_DIR:-$D/data/swebench100}
export LABELED_DAYTONA_DIR=$D/rl-env-cleaner/mimo_swe
export KEYS_FILE=$D/keys.env
export RUN_DIR=$D/runs/$RUN_NAME
export XDG_CACHE_HOME=$D/.cache
export UV_CACHE_DIR=/home/ray/.cache/uv  # the prewarmed cache; a fresh one re-fetches git deps for an hour
export VENV=${VENV-$D/venvs/megatron}  # prebuilt (ops/build_venvs.sh): workers skip the per-worker `uv run` (it outlasts the raylet registration timeout)
# torch 2.13 needs NCCL 2.29; the image's /usr/lib libnccl (2.28.9) shadows the wheel's. EFA libs for multi-node.
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
# Anyscale's raylet env points W&B at its hosted server (lila.wandb.io) and entity; ours is wandb.ai. Forwarded to workers.
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai} WANDB_ENTITY=${WANDB_ENTITY:-sky-posttraining-uc-berkeley}
ulimit -n 524288 || true
mkdir -p "$RUN_DIR"
# SANDBOX=dune: Dune Cloud (Daytona's API at DAYTONA_API_URL; dune_keys.env overrides keys.env's Daytona key). Dune has
# no preview URLs, so the sandbox relay polls the stubs through the exec API.
EXTRA_ARGS=()
if [[ "${SANDBOX:-daytona}" == dune ]]; then
  umask 077; cat $D/keys.env $D/dune_keys.env > "$RUN_DIR/keys.env"; umask 022
  export KEYS_FILE="$RUN_DIR/keys.env"
  EXTRA_ARGS+=(skycap.sandbox_relay.transport=exec)
fi
cd $D/SkyRL-skycap
exec bash examples/train_integrations/harbor_skycap/run_fully_async_qwen36_mini_swe.sh "${EXTRA_ARGS[@]}" "${@:2}"
