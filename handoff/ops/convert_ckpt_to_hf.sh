#!/usr/bin/env bash
# Convert a SkyRL Megatron checkpoint (torch_dist, any TP/EP) of Nemotron-3.5-Lightning-30B-A3B into an HF
# safetensors dir that `vllm serve` loads like the base model. CPU only, single process, no GPU / Ray / torchrun.
# ~1-3 min and ~15 GB RAM on a worker node; output ~66 GB (same shard layout/index/config/tokenizer as the base).
#
# Run ON THE NODE THAT HOLDS THE CHECKPOINT (paths are worker-local), e.g. from the head:
#   ssh -p 2222 <node> 'bash /mnt/local_storage/charlie/ops/convert_ckpt_to_hf.sh \
#       /mnt/local_storage/charlie/runs/nem35-mimo-fa-1016-0654/ckpt_keep/global_step_125'
# (copy this script + convert_ckpt_to_hf.py + validate_hf_export.py to $D/ops on the worker first.)
#
# Usage: convert_ckpt_to_hf.sh <global_step_dir> [out_dir] [extra converter flags, e.g. --drop-mtp]
#   default out_dir: $D/hf_export/<run_name>-step<N>
# Env: D (default /mnt/local_storage/charlie), BASE_HF, PYTHON (any python with torch>=2.4 + safetensors),
#      VALIDATE=1 (default) runs validate_hf_export.py against the base afterwards.
# Then:  vllm serve <out_dir> --served-model-name ... --trust-remote-code (same flags as for the base model).
set -euo pipefail

D="${D:-/mnt/local_storage/charlie}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${1:?usage: $0 <global_step_dir> [out_dir] [--drop-mtp]}"
shift
CKPT="$(cd "$CKPT" && pwd)"
STEP="$(basename "$CKPT")"; STEP="${STEP#global_step_}"
RUN="$(basename "$(dirname "$(dirname "$CKPT")")")"
OUT="$D/hf_export/${RUN}-step${STEP}"
if [[ $# -gt 0 && "$1" != --* ]]; then OUT="$1"; shift; fi
BASE_HF="${BASE_HF:-$D/hf/Nemotron-3.5-Lightning-30B-A3B}"
PYTHON="${PYTHON:-$D/venvs/megatron/bin/python}"

need_gb=80
mkdir -p "$(dirname "$OUT")"
avail_gb=$(df -BG --output=avail "$(dirname "$OUT")" | tail -1 | tr -dc 0-9)
(( avail_gb > need_gb )) || { echo "only ${avail_gb} GB free under $(dirname "$OUT"), need ~${need_gb}" >&2; exit 1; }

echo "ckpt=$CKPT"; echo "out=$OUT"; echo "base=$BASE_HF"
# No GPUs, no Ray: the converter is plain torch.distributed.checkpoint reads + safetensors writes.
# Cap CPU threads: the inference nodes' vLLM engines share these CPUs.
export CUDA_VISIBLE_DEVICES="" RAY_ADDRESS=local OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
"$PYTHON" -W ignore "$HERE/convert_ckpt_to_hf.py" --ckpt "$CKPT" --base-hf "$BASE_HF" --out "$OUT" "$@"

if [[ "${VALIDATE:-1}" == 1 ]]; then
  "$PYTHON" -W ignore "$HERE/validate_hf_export.py" --export "$OUT" --base-hf "$BASE_HF"
fi
echo "HF export ready: $OUT"
