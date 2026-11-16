#!/usr/bin/env bash
# Cluster wrapper for the repo's run_eval_skycap.sh: one SkyCap eval on THIS worker's 8 GPUs, pinned by a private Ray head.
#   bash ops/evals_1104/skycap_eval.sh <job> <nemotron|qwen> <swe|tb> [overrides...]   (env: SAMPLES, N_CONCURRENT, TASKS_DIR, ...)
set -euo pipefail
D=/mnt/local_storage/charlie
# Run from a snapshot: bash reads a script as it goes, so editing (or rsyncing) this file would break a running job.
if [[ -z "${EVAL_SNAPSHOT:-}" ]]; then
  mkdir -p $D/runs/$1; cp "$0" $D/runs/$1/launcher.sh
  EVAL_SNAPSHOT=1 exec bash $D/runs/$1/launcher.sh "$@"
fi
JOB=${1:?job}; FAM=${2:?nemotron or qwen}; BENCH=${3:?swe or tb}; shift 3
case "$FAM" in
  nemotron) export MODEL_FAMILY=nemotron-3.5 MODEL=$D/hf/Nemotron-3.5-Lightning-BF16 ;;
  qwen) export MODEL_FAMILY=qwen3.5 MODEL=$D/hf/Qwen3.5-9B ;;
  *) echo "model nemotron or qwen" >&2; exit 1 ;;
esac
case "$BENCH" in
  swe) export TASKS_DIR=${TASKS_DIR:-$D/data/swebench500std} ;;
  tb) export TASKS_DIR=${TASKS_DIR:-$D/data/tbench21nv} ;;
esac
export BENCH RUN_DIR=$D/runs/$JOB LABELS_OWNER=charlieruan PYTHON=$D/venvs/evals/bin/python
export UV_CACHE_DIR=/home/ray/.cache/uv XDG_CACHE_HOME=$D/.cache
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1 SKYRL_RAY_PG_TIMEOUT_IN_S=1800
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
set -a; . $D/keys.env; set +a
export HF_TOKEN=${HF_TOKEN:-$HF_API_KEY}
# SANDBOX=dune: Dune Cloud (Daytona's API at DAYTONA_API_URL) instead of Daytona. Dune's per-sandbox maximum is
# 2 vCPU / 4 GiB, so Terminal-Bench sandboxes are capped there (NVIDIA's recipe uses 4 / 8 GB).
if [[ "${SANDBOX:-daytona}" == dune ]]; then
  set -a; . $D/dune_keys.env; set +a
  export TB_CPUS=${TB_CPUS:-2} TB_MEMORY_MB=${TB_MEMORY_MB:-4096}
  # Large SWE-bench images (matplotlib) fail to build on Dune's default disk (buildctl exit 1): 30 GB (max 64).
  export STORAGE_MB=${STORAGE_MB:-30720}
  export RELAY_TRANSPORT=exec
fi
ulimit -n 524288 || true
mkdir -p $D/runs/$JOB
cp $D/SkyRL/examples/train_integrations/harbor_skycap/run_eval_skycap.sh $D/runs/$JOB/run_eval_skycap.sh
exec env RAY_BIN=$D/venvs/evals/bin/ray REPO=$D/SkyRL bash $D/ops/with_private_ray.sh \
  bash $D/runs/$JOB/run_eval_skycap.sh "$JOB" ${STORAGE_MB:+harbor_trial_config.environment.override_storage_mb=$STORAGE_MB} "$@"
