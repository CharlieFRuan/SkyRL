#!/usr/bin/env bash
# No-SkyCap baseline: stock `harbor run` (a JobConfig) against NUM_ENGINES plain `vllm serve` replicas on THIS worker,
# behind a prefix-affinity router. SWE (mini-swe-agent, in the sandbox) reaches the router through the same sandbox relay
# as the SkyCap arm (stub in the sandbox, polled by relay_manager_standalone.py); TB (Terminus-2, in harbor's process)
# calls the router directly. Same sandboxes, timeouts, resources and sampling as ops/evals_1104/skycap_eval.sh.
#   bash ops/evals_1104/baseline_eval.sh <job> <nemotron|qwen> <swe|tb>   (env: SAMPLES, N_CONCURRENT, TASKS_DIR, NUM_ENGINES)
set -euo pipefail
D=/mnt/local_storage/charlie
# Run from a snapshot: bash reads a script as it goes, so editing (or rsyncing) this file would break a running job.
if [[ -z "${EVAL_SNAPSHOT:-}" ]]; then
  mkdir -p $D/runs/$1; cp "$0" $D/runs/$1/launcher.sh
  EVAL_SNAPSHOT=1 exec bash $D/runs/$1/launcher.sh "$@"
fi
JOB=${1:?job}; FAM=${2:?nemotron or qwen}; BENCH=${3:?swe or tb}
SAMPLES=${SAMPLES:-3}
N_CONCURRENT=${N_CONCURRENT:-300}
# TP_SIZE: GPUs per replica (NUM_ENGINES defaults to 8 / TP_SIZE). ROUTER: affinity (rendezvous hash on the
# trajectory's prefix) or vllm-router (its cache_aware policy: prefix affinity, rebalanced past the thresholds below).
TP_SIZE=${TP_SIZE:-1}
NUM_ENGINES=${NUM_ENGINES:-$((8 / TP_SIZE))}
ROUTER=${ROUTER:-affinity}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
VENV=$D/venvs/evals
HERE=$D/SkyRL/examples/train_integrations/harbor_skycap
case "$FAM" in
  nemotron)
    MODEL=$D/hf/Nemotron-3.5-Lightning-BF16 SERVED=nemotron-3.5-lightning MSWEA=$HERE/mini_swe_agent_swebench_nemotron35.yaml
    SAMPLING='{"temperature": 1.0, "top_p": 0.95}'
    # Every earlier turn's thinking stays in the history, as the SkyCap arm (skycap.thinking_retention=all).
    CHAT_TEMPLATE_KWARGS='{"enable_thinking": true, "truncate_history_thinking": false}' TB_MAX_TURNS=${TB_MAX_TURNS:-1000000}
    # The model card's parsers.
    PARSERS=(--reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice)
    MODEL_FLAGS=() ;;
  qwen)
    MODEL=$D/hf/Qwen3.5-9B SERVED=qwen3.5-9b MSWEA=${MSWEA_CONFIG:-$HERE/mini_swe_agent_swebench_qwen35.yaml}
    SAMPLING='{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}'
    CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}' TB_MAX_TURNS=${TB_MAX_TURNS:-250}
    PARSERS=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice)
    MODEL_FLAGS=(--language-model-only --gdn-prefill-backend triton) ;;
  *) echo "model nemotron or qwen" >&2; exit 1 ;;
esac
case "$BENCH" in
  swe) TASKS_DIR=${TASKS_DIR:-$D/data/swebench500std} ;;
  # Terminus-2 replays `content`; like the SkyCap arm (skycap.use_raw_content), serve raw text (thinking inline).
  tb) TASKS_DIR=${TASKS_DIR:-$D/data/tbench21nv}; PARSERS=() ;;
  *) echo "bench swe or tb" >&2; exit 1 ;;
esac
RUN_DIR=$D/runs/$JOB
mkdir -p $RUN_DIR/vllm $RUN_DIR/jobs
NODE_IP=$(hostname -I | awk '{print $1}')
export UV_CACHE_DIR=/home/ray/.cache/uv XDG_CACHE_HOME=$D/.cache
export LD_LIBRARY_PATH=$D/nccl/pkg/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
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

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

: > $RUN_DIR/backends.txt
for i in $(seq 0 $((NUM_ENGINES - 1))); do
  GPUS=$(seq -s, $((i * TP_SIZE)) $((i * TP_SIZE + TP_SIZE - 1)))
  CUDA_VISIBLE_DEVICES=$GPUS $VENV/bin/vllm serve $MODEL --served-model-name $SERVED --host 127.0.0.1 --port $((8100 + i)) \
    --tensor-parallel-size $TP_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.9 --enable-prefix-caching "${PARSERS[@]}" "${MODEL_FLAGS[@]}" \
    ${SERVE_EXTRA:-} > $RUN_DIR/vllm/engine$i.log 2>&1 &
  PIDS+=($!)
  echo "http://127.0.0.1:$((8100 + i))" >> $RUN_DIR/backends.txt
done
for i in $(seq 0 $((NUM_ENGINES - 1))); do
  until curl -sf http://127.0.0.1:$((8100 + i))/health >/dev/null; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "engine $i died:"; tail -30 $RUN_DIR/vllm/engine$i.log; exit 1; }
    sleep 5
  done
done
echo "==> $NUM_ENGINES vLLM replicas up"
if [[ "$ROUTER" == vllm-router ]]; then
  $VENV/bin/python -m vllm_router.launch_router --host 127.0.0.1 --port 8000 --worker-urls $(cat $RUN_DIR/backends.txt) \
    --policy cache_aware --balance-abs-threshold ${BALANCE_ABS:-8} --balance-rel-threshold ${BALANCE_REL:-1.3} \
    --request-timeout-secs 7200 > $RUN_DIR/router.log 2>&1 &
else
  $VENV/bin/python $D/rl-env-cleaner/mimo_swe/affinity_router.py --backends $RUN_DIR/backends.txt --host 127.0.0.1 --port 8000 \
    > $RUN_DIR/router.log 2>&1 &
fi
PIDS+=($!)

RELAY=""
if [[ "$BENCH" == swe ]]; then
  rm -f $RUN_DIR/relay_url
  PYTHONPATH=$D/SkyRL $VENV/bin/python $D/ops/relay_manager_standalone.py $NODE_IP $RUN_DIR/relay_url > $RUN_DIR/relay.log 2>&1 &
  PIDS+=($!)
  until [ -s $RUN_DIR/relay_url ]; do sleep 1; done
  RELAY="$(cat $RUN_DIR/relay_url)"
  echo "==> relay manager $RELAY"
fi

JOB=$JOB RUN_DIR=$RUN_DIR BENCH=$BENCH TASKS_DIR=$TASKS_DIR SERVED=$SERVED MSWEA=$MSWEA SAMPLES=$SAMPLES \
N_CONCURRENT=$N_CONCURRENT RELAY=$RELAY SAMPLING="$SAMPLING" MAX_MODEL_LEN=$MAX_MODEL_LEN \
CHAT_TEMPLATE_KWARGS="$CHAT_TEMPLATE_KWARGS" TB_MAX_TURNS=$TB_MAX_TURNS \
  $VENV/bin/python $D/ops/evals_1104/make_job_config.py > $RUN_DIR/job.yaml
export PATH=$VENV/bin:$PATH PYTHONPATH=$D/SkyRL
export OPENAI_API_KEY=dummy MSWEA_API_KEY=dummy HOSTED_VLLM_API_KEY=dummy
harbor run -y -c $RUN_DIR/job.yaml
echo "==> done: $JOB"
