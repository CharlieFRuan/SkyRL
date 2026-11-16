#!/usr/bin/env bash
# Fresh-node setup for the 2026-11-10 timeout repro (inference only). Runs ON a worker; idempotent.
#   FULL=1: download model + data + build venv from scratch (first node). Others rsync from the first node.
set -uo pipefail
D=/mnt/local_storage/charlie; mkdir -p $D/{logs,runs,hf,data,tmp,venvs}
set -a; . $D/keys.env; set +a
export HF_TOKEN=${HF_TOKEN:-$HF_API_KEY} HF_HUB_ENABLE_HF_TRANSFER=0 UV_CACHE_DIR=/home/ray/.cache/uv UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=8
HF="uvx --from huggingface_hub[cli]==0.36.0 hf"
( [ -f $D/hf/Qwen3.5-9B/.done ] || { $HF download Qwen/Qwen3.5-9B --local-dir $D/hf/Qwen3.5-9B > $D/logs/hf_qwen35.log 2>&1 && touch $D/hf/Qwen3.5-9B/.done; }; echo "MODEL_DONE $?" ) &
( cd $D/data; if [ ! -f calibforge_v2/.done ]; then
    RDL_AUTH="Bearer $HF_TOKEN" python3 $D/ops/rdl.py https://huggingface.co/datasets/Fzz1/tb-full-data/resolve/main/data/tasks-calibforge_clean_v2.tar $D/data/cf_v2.tar 64 > $D/logs/cf_dl.log 2>&1 \
      && rm -rf cf_v2_x && mkdir cf_v2_x && tar xf cf_v2.tar -C cf_v2_x && echo extracted; fi; echo "DATA_DONE $?" ) &
[ -f $D/nccl/pkg/nvidia/nccl/lib/libnccl.so.2 ] || uv pip install --quiet --no-deps --target $D/nccl/pkg nvidia-nccl-cu13==2.29.7; echo "NCCL_DONE $?"
cd $D/SkyRL
for i in 1 2 3; do
  UV_PROJECT_ENVIRONMENT=$D/venvs/upstream-megatron timeout 5400 uv sync --frozen --extra megatron --extra harbor --extra skycap >> $D/logs/venv_upstream.log 2>&1 && { echo "VENV_DONE 0"; break; } || echo "venv attempt $i failed"
done
wait
echo SETUP_ALL_DONE
