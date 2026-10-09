#!/usr/bin/env bash
# Evaluation only: mini-swe-agent on a directory of Harbor tasks (e.g. 100 SWE-bench Verified tasks), with every
# model call going through skycap, which records each trial's exact tokens. No training.
#
#   RELAY_HOST=relay.example.com bash examples/train_integrations/harbor_skycap/run_swebench_eval_mini_swe.sh [overrides...]
#
# One node, 8 GPUs: 4 vLLM engines at TP 2 serving Qwen3.6-35B-A3B. mini-swe-agent runs inside each trial's Daytona
# sandbox (LabeledDaytonaEnv: labeled, ephemeral) and reaches skycap over HTTPS through a path-routing relay
# (relay_exposure.RelayPathExposure): server i at https://$RELAY_HOST/skycap/<this node's ip>/$((SKYCAP_PORT + i)).
# The relay must forward that prefix to <ip>:<port> with the prefix stripped (see relay_exposure.py for Caddy).
#
# Harbor writes each trial's directory under TRIALS_DIR, as `harbor run` would; the scores (stock Harbor's: the
# verifier's reward, 0 when it didn't run) go to $RUN_DIR/exports/eval/{results.jsonl,summary.json}, and skycap's
# records to $RUN_DIR/skycap. The stock baseline to compare with runs the same agent with the same config file:
#   harbor run -p $TASKS_DIR -a mini-swe-agent -m openai/$SERVED_MODEL_NAME --ak version=2.4.6 \
#     --ak config_file=$MSWEA_CONFIG -e daytona --environment-import-path labeled_daytona:LabeledDaytonaEnv ...
# with the agent and verifier timeouts below.
#
# Knobs (environment):
#   RELAY_HOST, SKYCAP_PORT   the relay's host name, and the first server's port (server i: +i)
#   NODE_IP                   this node's address as the relay reaches it (default: Ray's node address)
#   TASKS_DIR                 the Harbor task directories (default: the local SWE-bench Verified 100)
#   TRIALS_DIR                where Harbor writes trial directories (default $RUN_DIR/trials)
#   N_CONCURRENT              trials in flight (default 100); SAMPLES per task (default 1)
#   SKYCAP_SERVERS            skycap servers (default 4)
#   LABELED_DAYTONA_DIR       directory holding labeled_daytona.py, put on PYTHONPATH
#   KEYS_FILE                 file of KEY=value lines holding DAYTONA_API_KEY (if not already exported)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/examples/train_integrations/harbor_skycap"
cd "$REPO"

MODEL="${MODEL:-/mnt/local_storage/charlieruan/hf/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}"
TASKS_DIR="${TASKS_DIR:-/home/charlieruan/mimo/swebench100/raw}"
MSWEA_CONFIG="${MSWEA_CONFIG:-$HERE/mini_swe_agent_qwen36.yaml}"
MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.6}"
RELAY_HOST="${RELAY_HOST:?set RELAY_HOST to the public host name of the relay}"
SKYCAP_PORT="${SKYCAP_PORT:-11500}"
NODE_IP="${NODE_IP:-}"
SKYCAP_SERVERS="${SKYCAP_SERVERS:-4}"
N_CONCURRENT="${N_CONCURRENT:-100}"
SAMPLES="${SAMPLES:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-73728}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_ENGINES="${NUM_ENGINES:-4}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1800}"
VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-300}"
LABELED_DAYTONA_DIR="${LABELED_DAYTONA_DIR:-/home/charlieruan/mimo/rl-env-cleaner/mimo_swe}"
KEYS_FILE="${KEYS_FILE:-$HOME/charlie_keys.env}"

RUN_ID="$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")"
EXPERIMENT="${EXPERIMENT:-swebench-eval-skycap-$RUN_ID}"
RUN_DIR="${RUN_DIR:-/tmp/harbor/runs/$EXPERIMENT}"
TRIALS_DIR="${TRIALS_DIR:-$RUN_DIR/trials}"
mkdir -p "$RUN_DIR" "$TRIALS_DIR"

[[ -d "$TASKS_DIR" ]] || { echo "no tasks at $TASKS_DIR" >&2; exit 1; }
[[ -f "$LABELED_DAYTONA_DIR/labeled_daytona.py" ]] || { echo "no labeled_daytona.py in $LABELED_DAYTONA_DIR" >&2; exit 1; }
if [[ -z "${DAYTONA_API_KEY:-}" && -f "$KEYS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$KEYS_FILE"
  set +a
fi
: "${DAYTONA_API_KEY:?export DAYTONA_API_KEY or put it in KEYS_FILE}"

# Harbor imports the environment class in the Ray worker that runs the trials: export PYTHONPATH to Ray's workers.
export PYTHONPATH="$LABELED_DAYTONA_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SKYRL_PYTHONPATH_EXPORT=1
# One attempt per trial, as `harbor run` (the baseline) makes: a failed trial scores 0 instead of being rerun.
export SKYCAP_TRIAL_ATTEMPTS="${SKYCAP_TRIAL_ATTEMPTS:-1}"
# LabeledDaytonaEnv's sandbox label.
export MIMO_RUN_ID="${MIMO_RUN_ID:-$EXPERIMENT}"

EXPOSURE_ARGS=(
  skycap.exposure.type=examples.train_integrations.harbor_skycap.relay_exposure:RelayPathExposure
  skycap.exposure.kwargs.relay_host="$RELAY_HOST"
  skycap.exposure.kwargs.port="$SKYCAP_PORT"
)
if [[ -n "$NODE_IP" ]]; then
  EXPOSURE_ARGS+=(skycap.exposure.kwargs.node_ip="$NODE_IP")
fi

# A private Ray cluster for the run, even on a node that already has one.
export RAY_ADDRESS=local
export RAY_enable_ray_event=0
export RAY_task_events_report_interval_ms=0
unset RAY_enable_core_worker_ray_event_to_aggregator RAY_enable_export_api_write_config
unset RAY_DASHBOARD_AGGREGATOR_AGENT_EVENTS_EXPORT_ADDR RAY_DASHBOARD_AGGREGATOR_AGENT_PUBLISHER_MAX_RETRIES

echo "==> $EXPERIMENT: $(find "$TASKS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) tasks x $SAMPLES, trials in $TRIALS_DIR"
uv run --isolated --extra fsdp --extra harbor --extra skycap \
  -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap_eval \
  data.train_data="['$TASKS_DIR']" \
  data.val_data="['$TASKS_DIR']" \
  trainer.policy.model.path="$MODEL" \
  generator.inference_engine.served_model_name="$SERVED_MODEL_NAME" \
  trainer.project_name=harbor-skycap-eval \
  trainer.run_name="$EXPERIMENT" \
  trainer.logger=console \
  trainer.export_path="$RUN_DIR/exports" \
  trainer.ckpt_path="$RUN_DIR/ckpts" \
  trainer.log_path="$RUN_DIR/logs" \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.top_k=20 \
  generator.sampling_params.min_p=0.0 \
  generator.eval_n_samples_per_prompt="$SAMPLES" \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=2 \
  generator.rate_limit.max_concurrency="$N_CONCURRENT" \
  skycap.num_servers="$SKYCAP_SERVERS" \
  skycap.record_dir="$RUN_DIR/skycap" \
  skycap.renderer=qwen3.6 \
  skycap.chat_template_kwargs.enable_thinking=true \
  skycap.chat_template_kwargs.preserve_thinking=true \
  skycap.thinking_retention=all \
  skycap.max_tokens_policy=reject \
  "${EXPOSURE_ARGS[@]}" \
  harbor_trial_config.trials_dir="$TRIALS_DIR" \
  harbor_trial_config.agent.name=mini-swe-agent \
  harbor_trial_config.agent.kwargs.version="$MINI_SWE_AGENT_VERSION" \
  harbor_trial_config.agent.kwargs.config_file="$MSWEA_CONFIG" \
  harbor_trial_config.agent.override_timeout_sec="$AGENT_TIMEOUT" \
  harbor_trial_config.verifier.override_timeout_sec="$VERIFIER_TIMEOUT" \
  harbor_trial_config.environment.type=daytona \
  harbor_trial_config.environment.import_path=labeled_daytona:LabeledDaytonaEnv \
  harbor_trial_config.environment.override_cpus=null \
  harbor_trial_config.environment.override_memory_mb=null \
  harbor_trial_config.environment.override_storage_mb=null \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=0 \
  "$@" 2>&1 | tee -a "$RUN_DIR/run.log"

echo
echo "==> done: $EXPERIMENT"
echo "    scores:         $RUN_DIR/exports/eval/summary.json, results.jsonl"
echo "    Harbor trials:  $TRIALS_DIR"
echo "    skycap records: $RUN_DIR/skycap"
echo "    log:            $RUN_DIR/run.log"
